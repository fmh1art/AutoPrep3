"""
CodeAgentPlanMode — 规划-执行双阶段 Agent 框架。

阶段：
  1) 工作区扫描：规则式扫描容器内工作区，生成目录结构描述
  2) 规划：使用 planning agent 读取任务指令 + 工作区概览，生成 JSON 格式的实现计划
  3) 执行：将 JSON 计划序列化为文本，注入 execution agent 的初始 message，按计划执行
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time
import json
import re
import logging

from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2
from src.module.gpt_inference import SimpleAPICaller
from src.agent.code_agent import CodeAgent, AgentResult

logger = logging.getLogger(__name__)


def render_planning_prompt(
    instruction: str,
    workspace_overview: str,
    num_candidate_plans: int = 1,
    base_dir: Optional[str] = None,
) -> str:
    ctx: dict[str, Any] = {
        "instruction": instruction,
        "workspace_overview": workspace_overview,
    }
    if num_candidate_plans > 1:
        ctx["num_candidate_plans"] = num_candidate_plans
    return render_j2("code_agent_plan_mode_planning.j2", ctx, base_dir=base_dir)


def render_execution_prompt(
    instruction: str, repo_path: str, serialized_plan: str, base_dir: Optional[str] = None
) -> str:
    return render_j2(
        "code_agent_plan_mode_execution.j2",
        {"instruction": instruction, "repo_path": repo_path, "serialized_plan": serialized_plan},
        base_dir=base_dir,
    )


@dataclass
class ReactStepRecord:
    index: int
    role: str
    content_preview: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class CodeAgentPlanMode:

    def __init__(
        self,
        planner_cfg: dict | None = None,
        executor_llm: Any | None = None,
        ce_cfg: dict | None = None,
        executor_price: dict[str, float] | None = None,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
        planner_llm: Any | None = None,  # Backward compatibility
    ):
        if planner_cfg is None and planner_llm is not None:
            # Backward compatibility: old interface
            self.planner_caller = SimpleAPICaller(
                llm_name=getattr(planner_llm, "model", "").replace("openai/", ""),
                api_key=getattr(planner_llm, "api_key", ""),
                base_url=getattr(planner_llm, "base_url", None),
                api_version=getattr(planner_llm, "api_version", None),
            )
            self.executor_llm = executor_llm
            self.ce_cfg = None
            self.executor_price = {}
            self.tools = tools
            self.system_prompt_kwargs = system_prompt_kwargs or {}
        else:
            # New interface
            self.planner_caller = SimpleAPICaller(
                llm_name=planner_cfg["llm_name"],
                api_key=planner_cfg["key"],
                base_url=planner_cfg.get("openai_base_url"),
                api_version=planner_cfg.get("api_version"),
            )
            self.executor_llm = executor_llm
            self.ce_cfg = ce_cfg
            self.executor_price = executor_price or {}
            self.tools = tools
            self.system_prompt_kwargs = system_prompt_kwargs or {}

    # ----------------------------
    # Internal helpers (metrics)
    # ----------------------------

    @staticmethod
    def _merge_metrics_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Merge two metrics dicts by summing token usages and costs."""
        return {
            "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
            "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
            "reasoning_tokens": a.get("reasoning_tokens", 0) + b.get("reasoning_tokens", 0),
            "cache_read_tokens": a.get("cache_read_tokens", 0) + b.get("cache_read_tokens", 0),
            "cache_write_tokens": a.get("cache_write_tokens", 0) + b.get("cache_write_tokens", 0),
            "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
            "accumulated_cost": a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0),
            "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
            "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
        }

    # ----------------------------
    # Filesystem helpers
    # ----------------------------

    @staticmethod
    def _ensure_dirs(*paths: str) -> None:
        for p in paths:
            os.makedirs(p, exist_ok=True)

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ----------------------------
    # Workspace scanning
    # ----------------------------

    @staticmethod
    def scan_workspace(
        workspace: DockerWorkspace,
        repo_path: str = "/workspace",
        max_files: int = 150,
        max_files_per_dir: int = 30,
        timeout: float = 60.0,
    ) -> str:
        SCAN_SCRIPT = r"""
import os, json, sys

repo = sys.argv[1]
max_files = int(sys.argv[2])
max_per_dir = int(sys.argv[3])

BINARY_EXTS = {
    '.png','.jpg','.jpeg','.gif','.bmp','.ico','.svg','.webp',
    '.woff','.woff2','.ttf','.eot','.otf',
    '.pyc','.pyo','.so','.o','.a','.dll','.dylib','.exe',
    '.zip','.tar','.gz','.bz2','.xz','.7z','.rar',
    '.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx',
    '.mp3','.mp4','.avi','.mov','.wav','.flac',
    '.db','.sqlite','.pickle','.pkl','.npy','.npz',
    '.class','.jar','.war',
}

SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', '.tox', '.eggs',
    '.mypy_cache', '.pytest_cache', '.nox', 'dist', 'build',
    '.agents_tmp', '.venv', 'venv', 'env',
}

LOW_PRIORITY_DIRS = {
    'doc', 'docs', 'documentation', 'tutorials',
    'benchmarks', 'benchmark', 'perf', 'performance',
    'test', 'tests', 'testing', 'testdata',
    'scripts', 'ci',
    'assets', 'images', 'figures', 'plots',
    'data', 'dataset', 'datasets', 'fixtures'
}

def is_low_priority(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    return any(p in LOW_PRIORITY_DIRS for p in parts)

total_files_on_disk = 0
for dp, dns, fns in os.walk(repo):
    dns[:] = [d for d in dns if d not in SKIP_DIRS]
    total_files_on_disk += len(fns)

all_entries = []
for dirpath, dirnames, filenames in os.walk(repo):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        rel = os.path.relpath(fpath, repo)
        ext = os.path.splitext(fname)[1].lower()
        try:
            size = os.path.getsize(fpath)
        except OSError:
            size = -1
        is_binary = ext in BINARY_EXTS or size > 2*1024*1024
        lines = -1
        if not is_binary and size >= 0 and size <= 512*1024:
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    content = f.read()
                lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            except Exception:
                pass
        all_entries.append({
            'path': rel,
            'size': size,
            'lines': lines,
            'binary': is_binary,
            'low_priority': is_low_priority(rel),
        })

high = [e for e in all_entries if not e['low_priority']]
low = [e for e in all_entries if e['low_priority']]

dir_counts = {}
for e in all_entries:
    top = e['path'].replace('\\', '/').split('/')[0]
    if top not in dir_counts:
        dir_counts[top] = {'total': 0, 'source': 0}
    dir_counts[top]['total'] += 1
    if not e['low_priority']:
        dir_counts[top]['source'] += 1

print(json.dumps({
    "high_priority": high,
    "low_priority": low,
    "total_on_disk": total_files_on_disk,
    "dir_counts": dir_counts,
}))
"""
        cmd = (
            f"python3 -c {shlex.quote(SCAN_SCRIPT)} "
            f"{shlex.quote(repo_path)} {max_files} {max_files_per_dir}"
        )
        try:
            result = workspace.execute_command(cmd, timeout=timeout)
            if result.exit_code != 0:
                logger.warning(f"Workspace scan failed: {getattr(result, 'stderr', '')}")
                return f"(workspace scan failed, working directory: {repo_path})"

            scan_data = json.loads(result.stdout.strip())
            high_priority = scan_data["high_priority"]
            low_priority = scan_data["low_priority"]
            total_on_disk = scan_data["total_on_disk"]
            dir_counts = scan_data.get("dir_counts", {})
        except Exception as e:
            logger.warning(f"Workspace scan error: {e}")
            return f"(workspace scan error: {e}, working directory: {repo_path})"

        out: list[str] = [f"Workspace root: {repo_path}"]
        out.append(f"Total files: {total_on_disk}")

        sorted_dirs = sorted(dir_counts.items(), key=lambda x: x[1]['source'], reverse=True)
        out.append("")
        out.append("Directory structure (source/total files):")
        for dir_name, counts in sorted_dirs[:20]:
            src = counts['source']
            tot = counts['total']
            if src > 0:
                out.append(f"  {dir_name}/  ({src} source, {tot} total)")
            else:
                out.append(f"  {dir_name}/  ({tot} files, non-source)")
        if len(sorted_dirs) > 20:
            out.append(f"  ... and {len(sorted_dirs) - 20} more directories")

        out.append("")
        out.append("Source files (most relevant for code changes):")
        out.append(f"{'Path':<62} {'Lines':>6}")
        out.append("-" * 70)

        shown = 0
        current_dir = None
        for e in high_priority:
            if shown >= max_files:
                remaining = len(high_priority) - shown
                if remaining > 0:
                    out.append(f"  ... ({remaining} more source files omitted)")
                break
            entry_dir = os.path.dirname(e["path"]) or "."
            if entry_dir != current_dir:
                current_dir = entry_dir
            lines_str = str(e["lines"]) if e["lines"] >= 0 else "-"
            path_display = e["path"]
            if len(path_display) > 60:
                path_display = "..." + path_display[-57:]
            out.append(f"{path_display:<62} {lines_str:>6}")
            shown += 1

        if low_priority:
            out.append("")
            out.append(f"Other files ({len(low_priority)} total, showing up to {max_files_per_dir}):")
            out.append(f"{'Path':<62} {'Lines':>6}")
            out.append("-" * 70)
            shown_lp = 0
            for e in low_priority:
                if shown_lp >= max_files_per_dir:
                    out.append(f"  ... ({len(low_priority) - shown_lp} more omitted)")
                    break
                lines_str = str(e["lines"]) if e["lines"] >= 0 else "-"
                path_display = e["path"]
                if len(path_display) > 60:
                    path_display = "..." + path_display[-57:]
                out.append(f"{path_display:<62} {lines_str:>6}")
                shown_lp += 1

        return "\n".join(out)

    # ----------------------------
    # JSON plan parsing & serialization
    # ----------------------------

    @staticmethod
    def _extract_plans_from_text(text: str, multi: bool = False):
        if not text:
            return None

        pattern = r"```json\s*(\[.*?\])\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(parsed, list) or len(parsed) == 0:
                continue

            if multi:
                if isinstance(parsed[0], list):
                    valid = all(
                        isinstance(p, list) and len(p) > 0 and all(isinstance(s, str) for s in p)
                        for p in parsed
                    )
                    if valid:
                        return parsed
                elif isinstance(parsed[0], str):
                    return [parsed]
            else:
                if isinstance(parsed[0], str):
                    return parsed
                elif isinstance(parsed[0], list):
                    return parsed[0] if len(parsed[0]) > 0 and isinstance(parsed[0][0], str) else None

        if not multi:
            try:
                parsed = json.loads(text.strip())
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], str):
                    return parsed
            except Exception:
                pass

        return None

    @staticmethod
    def _serialize_plan_for_execution(plan: list[str]) -> str:
        lines: list[str] = [f"### Subtasks ({len(plan)} total)", ""]
        for i, subtask in enumerate(plan, 1):
            lines.append(f"{i}. {subtask}")
        return "\n".join(lines)

    # ----------------------------
    # Cost estimation helpers
    # ----------------------------

    @staticmethod
    def _plan_to_ce_subtasks(plan: list[str]) -> list[dict]:
        return [{"non_negative_idx": i + 1, "title": s} for i, s in enumerate(plan)]

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        planner_trajectory_path: str | None = None,
        execution_trajectory_path: str | None = None,
        planner_meta: dict | None = None,
        execution_meta: dict | None = None,
        max_scan_files: int = 300,
        num_candidate_plans: int = 1,
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        logs_dir = os.path.join(out_dir, "log")
        swe_dir = os.path.join(out_dir, "swe_eval_logs")
        self._ensure_dirs(logs_dir, swe_dir)
        planner_traj = planner_trajectory_path or os.path.join(
            logs_dir, "planner_trajectory.md"
        )
        execution_traj = execution_trajectory_path or os.path.join(
            logs_dir, "execution_trajectory.md"
        )

        # ----------------
        # Phase 0: Workspace Scan
        # ----------------
        logger.info(f"Scanning workspace at {repo_path} (max {max_scan_files} files)...")
        workspace_overview = self.scan_workspace(
            workspace=workspace,
            repo_path=repo_path,
            max_files=max_scan_files,
        )
        logger.info(f"Workspace scan complete, overview length: {len(workspace_overview)} chars")
        self._write_text(os.path.join(logs_dir, "workspace_overview.txt"), workspace_overview)

        # ----------------
        # Phase 1: Planning (JSON) via SimpleAPICaller
        # ----------------
        use_ce = self.ce_cfg is not None and num_candidate_plans > 1
        effective_num_plans = num_candidate_plans if use_ce else 1

        planning_prompt = render_planning_prompt(
            instruction=instruction,
            workspace_overview=workspace_overview,
            num_candidate_plans=effective_num_plans,
        )

        try:
            with open(planner_traj, "w", encoding="utf-8") as f:
                f.write("# Planner Trajectory\n\n")
                if planner_meta:
                    f.write("## Agent Params (Planner)\n\n")
                    f.write(
                        "```json\n"
                        + json.dumps(planner_meta, ensure_ascii=False, indent=2)
                        + "\n```\n\n"
                    )
                f.write("## Prompt\n\n")
                f.write("```\n" + planning_prompt + "```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init planner trajectory file {planner_traj}: {e}")

        logger.info("Calling planner LLM via SimpleAPICaller...")
        max_plan_attempts = 3
        planner_messages = [{"role": "user", "content": planning_prompt}]
        parsed_plans = None
        usage_before = self.planner_caller.get_total_usage()

        for attempt in range(1, max_plan_attempts + 1):
            raw_response = self.planner_caller.chat(planner_messages)
            planner_usage = self.planner_caller.get_last_usage()
            logger.info(
                f"Planner attempt {attempt}: response {len(raw_response)} chars, usage: {planner_usage}"
            )

            try:
                with open(planner_traj, "a", encoding="utf-8") as f:
                    f.write(f"## Attempt {attempt}\n\n")
                    f.write(raw_response + "\n\n")
                    f.write(
                        "```json\n"
                        + json.dumps(planner_usage, ensure_ascii=False, indent=2, default=str)
                        + "\n```\n\n---\n\n"
                    )
            except Exception as e:
                logger.warning(f"Failed to append planner trajectory: {e}")

            if use_ce:
                parsed_plans = self._extract_plans_from_text(raw_response, multi=True)
                if parsed_plans is None or len(parsed_plans) == 0:
                    single = self._extract_plans_from_text(raw_response, multi=False)
                    parsed_plans = [single] if single else None
            else:
                single = self._extract_plans_from_text(raw_response, multi=False)
                parsed_plans = [single] if single else None

            if parsed_plans is not None:
                break

            if attempt < max_plan_attempts:
                retry_msg = (
                    "Your previous response could not be parsed as valid JSON. "
                    "Please output the plan again as a valid JSON code block (```json ... ```).\n"
                    f"Expected format: {'a JSON array of arrays (each inner array is a list of subtask strings)' if use_ce else 'a JSON array of strings (each string is one subtask)'}.\n"
                    "Do NOT include anything outside the JSON code block."
                )
                planner_messages.append({"role": "assistant", "content": raw_response})
                planner_messages.append({"role": "user", "content": retry_msg})
                logger.warning(
                    f"Planner attempt {attempt} failed to produce valid JSON, retrying..."
                )

        usage_after = self.planner_caller.get_total_usage()
        planner_metrics: dict[str, Any] = {
            "prompt_tokens": usage_after.get("input_tokens", 0) - usage_before.get("input_tokens", 0),
            "completion_tokens": usage_after.get("output_tokens", 0) - usage_before.get("output_tokens", 0),
            "cache_read_tokens": usage_after.get("cached_tokens", 0) - usage_before.get("cached_tokens", 0),
            "cache_write_tokens": 0,
            "reasoning_tokens": usage_after.get("reasoning_tokens", 0) - usage_before.get("reasoning_tokens", 0),
            "total_tokens": usage_after.get("total_tokens", 0) - usage_before.get("total_tokens", 0),
            "accumulated_cost": 0.0,
        }

        # ----------------
        # Extract plans from planner text response
        # ----------------
        ce_metrics: dict[str, Any] = {}
        ce_all_results: dict[str, Any] = {}

        if use_ce:
            candidate_plans = parsed_plans
            if candidate_plans is None:
                logger.warning("Failed to extract any plans after retries, using fallback")
                candidate_plans = [
                    ["Analyze the task instruction and implement the required changes"]
                ]

            logger.info(f"Extracted {len(candidate_plans)} candidate plans")
            self._write_json(os.path.join(logs_dir, "candidate_plans.json"), candidate_plans)

            # ----------------
            # Phase 1.5: Cost Estimation (via SimpleAPICaller)
            # ----------------
            from src.agent.cost_estimate import CEAgent

            plan_costs: list[dict[str, Any]] = []
            all_ce_metrics_list: list[dict[str, Any]] = []

            for plan_idx, plan in enumerate(candidate_plans):
                logger.info(f"Estimating cost for plan {plan_idx} ({len(plan)} subtasks)...")
                ce_agent = CEAgent(
                    ce_cfg=self.ce_cfg,
                    executor_price=self.executor_price,
                )
                ce_subtasks = self._plan_to_ce_subtasks(plan)
                try:
                    ce_output = ce_agent.estimate_cost(
                        instruction=instruction,
                        subtasks=ce_subtasks,
                    )
                    ce_results_data = ce_output["ce_results"]
                    estimated_cost = ce_agent.compute_plan_cost(ce_results_data)
                    estimated_uncertainty = ce_agent.compute_plan_uncertainty(ce_results_data)
                    result = {
                        "plan_index": plan_idx,
                        "plan": plan,
                        "estimated_dollar_cost": estimated_cost,
                        "estimated_uncertainty": estimated_uncertainty,
                        "ce_results": ce_results_data,
                        "ce_metrics": ce_output["metrics"],
                    }
                    logger.info(
                        f"Plan {plan_idx}: estimated_dollar_cost={estimated_cost}, uncertainty={estimated_uncertainty}"
                    )
                except Exception as e:
                    logger.error(f"Cost estimation failed for plan {plan_idx}: {e}")
                    result = {
                        "plan_index": plan_idx,
                        "plan": plan,
                        "estimated_dollar_cost": float("inf"),
                        "estimated_uncertainty": 1.0,
                        "ce_results": [],
                        "ce_metrics": {},
                        "error": str(e),
                    }
                plan_costs.append(result)
                if result.get("ce_metrics"):
                    all_ce_metrics_list.append(result["ce_metrics"])

            plan_costs.sort(key=lambda x: x["plan_index"])

            self._write_json(os.path.join(logs_dir, "plan_costs.json"), plan_costs)

            valid_costs = [pc for pc in plan_costs if pc["estimated_dollar_cost"] != float("inf")]
            if valid_costs:
                costs = [pc["estimated_dollar_cost"] for pc in valid_costs]
                min_cost = min(costs)

                for pc in valid_costs:
                    relative_cost = pc["estimated_dollar_cost"] / max(min_cost, 1e-9)
                    uncertainty = pc["estimated_uncertainty"]

                    cost_weight = float(os.getenv("CE_COST_WEIGHT", "0.7"))
                    uncertainty_weight = float(os.getenv("CE_UNCERTAINTY_WEIGHT", "0.3"))

                    cost_score = (relative_cost - 1.0) * cost_weight
                    uncertainty_score = uncertainty * uncertainty_weight

                    pc["score"] = cost_score + uncertainty_score
                    pc["relative_cost"] = relative_cost
                    logger.info(
                        f"Plan {pc['plan_index']}: "
                        f"cost={pc['estimated_dollar_cost']:.4f} "
                        f"(x{relative_cost:.2f}), "
                        f"uncertainty={uncertainty:.3f}, "
                        f"score={pc['score']:.3f}"
                    )

                best = min(valid_costs, key=lambda x: x["score"])
            else:
                best = plan_costs[0]

            json_plan = best["plan"]
            selected_plan_idx = best["plan_index"]
            logger.info(
                f"Selected plan {selected_plan_idx} with cost={best['estimated_dollar_cost']}, uncertainty={best.get('estimated_uncertainty', 'N/A')}"
            )

            ce_all_results = {
                "candidate_plans_count": len(candidate_plans),
                "selected_plan_index": selected_plan_idx,
                "plan_costs": plan_costs,
            }

            merged_ce = {}
            for m in all_ce_metrics_list:
                if m:
                    if not merged_ce:
                        merged_ce = dict(m)
                    else:
                        for k in (
                            "prompt_tokens",
                            "completion_tokens",
                            "reasoning_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                            "total_tokens",
                        ):
                            merged_ce[k] = merged_ce.get(k, 0) + m.get(k, 0)
                        merged_ce["accumulated_cost"] = merged_ce.get(
                            "accumulated_cost", 0.0
                        ) + m.get("accumulated_cost", 0.0)
            ce_metrics = merged_ce

        else:
            json_plan = parsed_plans[0] if parsed_plans else None

            if json_plan is None:
                logger.warning("Failed to extract JSON plan after retries, using fallback")
                json_plan = ["Analyze the task instruction and implement the required changes"]

        self._write_json(os.path.join(logs_dir, "plan.json"), json_plan)
        serialized_plan = self._serialize_plan_for_execution(json_plan)
        self._write_text(os.path.join(logs_dir, "plan_serialized.md"), serialized_plan)
        logger.info(f"Plan extracted: {len(json_plan)} subtasks")

        # ----------------
        # Phase 2: Execution (使用自定义 CodeAgent)
        # ----------------
        # 构建执行 agent 的 LLM 配置
        # 优先使用 executor_llm 的配置，否则使用 planner 的配置
        if self.executor_llm is not None:
            executor_cfg = {
                "llm_name": getattr(self.executor_llm, "model", "").replace("openai/", ""),
                "key": getattr(self.executor_llm, "api_key", ""),
                "openai_base_url": getattr(self.executor_llm, "base_url", None),
                "api_version": getattr(self.executor_llm, "api_version", None),
            }
        else:
            # 使用 planner 的配置
            executor_cfg = {
                "llm_name": self.planner_caller.llm_name,
                "key": self.planner_caller.api_key,
                "openai_base_url": self.planner_caller.base_url,
                "api_version": self.planner_caller.api_version,
            }

        executor_agent = CodeAgent(
            llm_cfg=executor_cfg,
            max_steps=100,
            max_retries_per_call=3,
        )

        execution_prompt = render_execution_prompt(
            instruction=instruction,
            repo_path=repo_path,
            serialized_plan=serialized_plan,
        )

        # 初始化执行轨迹记录
        execution_steps: list[ReactStepRecord] = []
        steps_json: list[dict[str, Any]] = []

        try:
            with open(execution_traj, "w", encoding="utf-8") as f:
                f.write("# Execution Trajectory\n\n")
                if execution_meta:
                    f.write("## Agent Params (Execution)\n\n")
                    f.write(
                        "```json\n"
                        + json.dumps(execution_meta, ensure_ascii=False, indent=2)
                        + "\n```\n\n"
                    )
                f.write("## Initial Prompt\n\n")
                f.write("```\n" + execution_prompt + "```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init execution trajectory file {execution_traj}: {e}")

        # 自定义回调函数来记录执行轨迹
        step_index = 0

        def exec_callback(record: dict):
            nonlocal step_index
            timestamp = time.time()
            rec = ReactStepRecord(
                index=step_index,
                role=record.get("role", ""),
                content_preview=record.get("observation", "")[:500],
                metrics=record.get("usage", {}),
                timestamp=timestamp,
                event_type=record.get("role", ""),
                payload=record,
            )
            execution_steps.append(rec)

            # 持久化到 trajectory.md
            try:
                with open(execution_traj, "a", encoding="utf-8") as f:
                    role = record.get("role", "")
                    if role == "assistant":
                        f.write(f"## Step {step_index} - Assistant\n")
                        thinking = record.get("thinking", "")
                        if thinking:
                            f.write(f"**Thought**: {thinking}\n\n")
                    elif role == "tool":
                        f.write(f"## Step {step_index} - Tool\n")
                        f.write(f"**Tool**: {record.get('tool_name', '')}\n")
                        args = record.get("tool_args", {})
                        if args:
                            f.write(f"**Args**: ```json\n{json.dumps(args, indent=2)}\n```\n\n")
                        obs = record.get("observation", "")
                        if obs:
                            f.write(f"**Observation**: ```\n{obs[:2000]}\n```\n\n")
                        usage = record.get("usage", {})
                        if usage:
                            f.write(
                                "**Metrics**: ```json\n"
                                + json.dumps(usage, ensure_ascii=False, indent=2)
                                + "\n```\n\n"
                            )
                    f.write("---\n\n")

                # 收集结构化步骤信息（JSON）
                step_payload = {
                    "index": step_index,
                    "role": role,
                    "timestamp": timestamp,
                    **record,
                }
                steps_json.append(step_payload)
            except Exception as ex:
                logger.warning(f"Error saving execution trajectory: {ex}")

            step_index += 1

        # 运行执行 agent
        exec_result = executor_agent.run(
            instruction=execution_prompt,
            workspace=workspace,
            callbacks=[exec_callback] + (callbacks or []),
            output_dir=logs_dir,
        )
        exec_metrics = exec_result.metrics

        # 写出结构化的步骤日志（JSON）
        try:
            self._write_json(
                os.path.join(logs_dir, "execution_steps.json"),
                {"initial_prompt": execution_prompt, "steps": steps_json},
            )
        except Exception as e:
            logger.warning(f"Failed to write execution_steps.json: {e}")

        # ----------------
        # 汇总与产物输出
        # ----------------
        total_metrics = self._merge_metrics_dict(planner_metrics, exec_metrics)
        token_report: dict[str, Any] = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "total": total_metrics,
        }
        if ce_metrics:
            token_report["cost_estimation"] = ce_metrics

        self._write_json(os.path.join(logs_dir, "token_usage.json"), token_report)

        def _summarize_usage(m: dict[str, Any]) -> dict[str, int]:
            return {
                "prompt_tokens": int(m.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(m.get("completion_tokens", 0) or 0),
                "answer_tokens": int(m.get("completion_tokens", 0) or 0),
                "cache_read_tokens": int(m.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(m.get("cache_write_tokens", 0) or 0),
                "reasoning_tokens": int(m.get("reasoning_tokens", 0) or 0),
            }

        results_payload: dict[str, Any] = {
            "planner_summary": _summarize_usage(planner_metrics),
            "execution_summary": _summarize_usage(exec_metrics),
            "total_summary": _summarize_usage(total_metrics),
        }
        if ce_metrics:
            results_payload["cost_estimation_summary"] = _summarize_usage(ce_metrics)
        if ce_all_results:
            results_payload["cost_estimation_details"] = ce_all_results
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [
            f"# CodeAgentPlanMode Run Log",
            "",
            f"- Timestamp: {now}",
            f"- Plan subtasks: {len(json_plan)}",
            f"- Cost estimation enabled: {use_ce}",
            f"- Logs Dir: `{logs_dir}`",
            f"- SWE Eval Dir: `{swe_dir}`",
            "",
            "## Metrics Summary",
            f"- Planner accumulated_cost: {planner_metrics.get('accumulated_cost', 0.0)}",
            f"- Executor accumulated_cost: {exec_metrics.get('accumulated_cost', 0.0)}",
        ]
        if ce_metrics:
            log_lines.append(
                f"- CostEstimation accumulated_cost: {ce_metrics.get('accumulated_cost', 0.0)}"
            )
            log_lines.append(f"- CostEstimation tokens: {_summarize_usage(ce_metrics)}")
        log_lines.extend(
            [
                f"- Planner tokens: {_summarize_usage(planner_metrics)}",
                f"- Executor tokens: {_summarize_usage(exec_metrics)}",
                f"- Total tokens: {_summarize_usage(total_metrics)}",
            ]
        )
        self._write_text(os.path.join(logs_dir, "log.md"), "\n".join(log_lines))

        all_metrics: dict[str, Any] = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "total": total_metrics,
        }
        if ce_metrics:
            all_metrics["cost_estimation"] = ce_metrics

        other: dict[str, Any] = {
            "logs_dir": logs_dir,
            "swe_eval_logs": swe_dir,
            "plan_json": json_plan,
        }
        if ce_all_results:
            other["cost_estimation_details"] = ce_all_results

        result = AgentResult(
            metrics=all_metrics,
            conversation=None,
        )
        # 添加额外属性以兼容旧接口
        result.execution_trajectory = execution_steps
        result.execution_steps_metrics = []
        result.other_content = other

        return result
