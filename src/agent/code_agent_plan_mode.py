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

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.llm import content_to_str
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2
from src.module.gpt_inference import SimpleAPICaller

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

def render_execution_prompt(instruction: str, repo_path: str, serialized_plan: str, base_dir: Optional[str] = None) -> str:
    return render_j2(
        "code_agent_plan_mode_execution.j2",
        {"instruction": instruction, "repo_path": repo_path, "serialized_plan": serialized_plan},
        base_dir=base_dir,
    )

from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response

logger = get_logger(__name__)


@dataclass
class ReactStepRecord:
    index: int
    role: str
    content_preview: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None
    execution_trajectory: list[ReactStepRecord] = field(default_factory=list)
    execution_steps_metrics: list[dict[str, Any]] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)

class CodeAgentPlanMode:

    def __init__(
        self,
        planner_cfg: dict | None = None,
        executor_llm: LLM | None = None,
        ce_cfg: dict | None = None,
        executor_price: dict[str, float] | None = None,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
        planner_llm: LLM | None = None,  # Backward compatibility
    ):
        if planner_cfg is None and planner_llm is not None:
            # Backward compatibility: old interface (planner_llm, executor_llm, tools)
            self.planner_caller = SimpleAPICaller(
                llm_name=planner_llm.model.replace("openai/", ""),
                api_key=getattr(planner_llm, "api_key", ""),
                base_url=getattr(planner_llm, "base_url", None),
                api_version=getattr(planner_llm, "api_version", None),
            )
            self.executor_llm = executor_llm
            self.ce_cfg = None
            self.executor_price = {}
            self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
            self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}
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
            self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
            self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}

    # ----------------------------
    # Internal helpers (metrics)
    # ----------------------------

    @staticmethod
    def _conversation_metrics(conversation: Conversation) -> dict[str, Any]:
        """Read metrics from conversation_stats so remote conversations report usage."""
        return conversation.conversation_stats.get_combined_metrics().get()

    @staticmethod
    def _merge_metrics_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Merge two metrics.get() dicts by summing token usages and costs.

        - accumulated_cost: sum
        - token_usages list: concatenate
        - accumulated_token_usage: element-wise sum
        - response_latencies/costs: concatenate
        - context_window: keep max when summing accumulated_token_usage
        """
        def sum_usage(u1: dict | None, u2: dict | None) -> dict | None:
            if not u1:
                return u2
            if not u2:
                return u1
            # 允许新增 answer_tokens（等同于 completion_tokens，用于与 reasoning_tokens 并列展示）
            comp_sum = u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0)
            reas_sum = u1.get("reasoning_tokens", 0) + u2.get("reasoning_tokens", 0)
            return {
                "model": u1.get("model", ""),
                "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
                "completion_tokens": comp_sum,
                "cache_read_tokens": u1.get("cache_read_tokens", 0) + u2.get("cache_read_tokens", 0),
                "cache_write_tokens": u1.get("cache_write_tokens", 0) + u2.get("cache_write_tokens", 0),
                "reasoning_tokens": reas_sum,
                "answer_tokens": comp_sum,
                "context_window": max(u1.get("context_window", 0), u2.get("context_window", 0)),
                "per_turn_token": 0,
                "response_id": "",
            }

        return {
            "accumulated_cost": (a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0)),
            "max_budget_per_task": a.get("max_budget_per_task") or b.get("max_budget_per_task"),
            "accumulated_token_usage": sum_usage(a.get("accumulated_token_usage"), b.get("accumulated_token_usage")),
            "costs": list(a.get("costs", [])) + list(b.get("costs", [])),
            "response_latencies": list(a.get("response_latencies", [])) + list(b.get("response_latencies", [])),
            "token_usages": list(a.get("token_usages", [])) + list(b.get("token_usages", [])),
        }

    # ----------------------------
    # Internal helpers (prompting)
    # ----------------------------
    # 提示渲染已抽取到 src/tools/funcs.py 中，其他模块可直接复用。

    # ----------------------------
    # Internal helpers (trajectory)
    # ----------------------------

    @staticmethod
    def _event_preview(event: Any) -> str:
        try:
            return str(event)
        except Exception:
            return repr(event)

    @staticmethod
    def _build_trajectory_md(events: list[Any]) -> str:
        """Build a simple markdown transcript of events for trajectory logging."""
        lines: list[str] = ["# Trajectory", ""]
        for idx, e in enumerate(events):
            ts = getattr(e, "timestamp", "")
            etype = e.__class__.__name__
            lines.append(f"## Step {idx+1} — {etype} @ {ts}")
            lines.append("")
            lines.append(CodeAgentPlanMode._event_preview(e))
            lines.append("")
        return "\n".join(lines)

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
        max_files: int = 300,
        max_files_per_dir: int = 50,
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

total_files_on_disk = 0
for dp, dns, fns in os.walk(repo):
    dns[:] = [d for d in dns if d not in SKIP_DIRS]
    total_files_on_disk += len(fns)

use_per_dir_limit = total_files_on_disk > max_files

entries = []
skipped_dirs = {}
for dirpath, dirnames, filenames in os.walk(repo):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    dir_count = 0
    dir_total = len(filenames)
    for fname in filenames:
        if use_per_dir_limit and dir_count >= max_per_dir:
            rel_dir = os.path.relpath(dirpath, repo)
            skipped_dirs[rel_dir] = dir_total - dir_count
            break
        fpath = os.path.join(dirpath, fname)
        rel = os.path.relpath(fpath, repo)
        ext = os.path.splitext(fname)[1].lower()
        try:
            size = os.path.getsize(fpath)
        except OSError:
            size = -1
        is_binary = ext in BINARY_EXTS or size > 2*1024*1024
        lines = -1
        tokens = -1
        if not is_binary and size >= 0 and size <= 512*1024:
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    content = f.read()
                lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
                tokens = len(content) // 4
            except Exception:
                pass
        entries.append({
            'path': rel,
            'size': size,
            'lines': lines,
            'tokens': tokens,
            'binary': is_binary,
        })
        dir_count += 1
        if len(entries) >= max_files:
            if dir_count < dir_total:
                rel_dir = os.path.relpath(dirpath, repo)
                skipped_dirs[rel_dir] = dir_total - dir_count
            break
    if len(entries) >= max_files:
        break

print(json.dumps({"entries": entries, "total_on_disk": total_files_on_disk, "skipped_dirs": skipped_dirs}))
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
            entries = scan_data["entries"]
            total_on_disk = scan_data["total_on_disk"]
            skipped_dirs = scan_data.get("skipped_dirs", {})
        except Exception as e:
            logger.warning(f"Workspace scan error: {e}")
            return f"(workspace scan error: {e}, working directory: {repo_path})"

        lines_out: list[str] = [f"Workspace root: {repo_path}"]
        lines_out.append(f"Total files on disk: {total_on_disk}")
        lines_out.append(f"Files listed: {len(entries)}")
        if total_on_disk > max_files:
            lines_out.append(f"(per-directory cap: {max_files_per_dir} files)")
        lines_out.append("")
        lines_out.append(f"{'Path':<60} {'Size':>8} {'Lines':>7} {'Tokens':>8} {'Type':>6}")
        lines_out.append("-" * 95)

        current_dir = None
        for e in entries:
            entry_dir = os.path.dirname(e['path']) or "."
            if entry_dir != current_dir:
                current_dir = entry_dir
                skipped_count = skipped_dirs.get(current_dir, 0)
                if skipped_count > 0:
                    if current_dir != ".":
                        lines_out.append(f"  ... ({skipped_count} more files in {current_dir}/ omitted)")

            size_str = f"{e['size']}B" if e['size'] >= 0 else "?"
            lines_str = str(e['lines']) if e['lines'] >= 0 else "-"
            tokens_str = str(e['tokens']) if e['tokens'] >= 0 else "-"
            type_str = "binary" if e['binary'] else "text"
            path_display = e['path']
            if len(path_display) > 58:
                path_display = "..." + path_display[-55:]
            lines_out.append(
                f"{path_display:<60} {size_str:>8} {lines_str:>7} {tokens_str:>8} {type_str:>6}"
            )

        for sd, cnt in skipped_dirs.items():
            if cnt > 0 and sd not in {os.path.dirname(e['path']) or "." for e in entries}:
                lines_out.append(f"  ... ({cnt} more files in {sd}/ omitted)")

        if len(entries) >= max_files:
            lines_out.append(f"\n(truncated at {max_files} files, {total_on_disk} total on disk)")

        return "\n".join(lines_out)

    # ----------------------------
    # JSON plan parsing & serialization
    # ----------------------------

    @staticmethod
    def _extract_plans_from_text(text: str, multi: bool = False):
        if not text:
            return None

        pattern = r'```json\s*(\[.*?\])\s*```'
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
        planner_traj = planner_trajectory_path or os.path.join(logs_dir, "planner_trajectory.md")
        execution_traj = execution_trajectory_path or os.path.join(logs_dir, "execution_trajectory.md")

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
                    f.write("```json\n" + json.dumps(planner_meta, ensure_ascii=False, indent=2) + "\n```\n\n")
                f.write("## Prompt\n\n")
                f.write("```\n" + planning_prompt + "\n```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init planner trajectory file {planner_traj}: {e}")

        logger.info("Calling planner LLM via SimpleAPICaller...")
        max_plan_attempts = 3
        planner_messages = [{"role": "user", "content": planning_prompt}]
        planner_response = None
        parsed_plans = None
        usage_before = self.planner_caller.get_total_usage()

        for attempt in range(1, max_plan_attempts + 1):
            raw_response = self.planner_caller.chat(planner_messages)
            planner_usage = self.planner_caller.get_last_usage()
            logger.info(f"Planner attempt {attempt}: response {len(raw_response)} chars, usage: {planner_usage}")

            try:
                with open(planner_traj, "a", encoding="utf-8") as f:
                    f.write(f"## Attempt {attempt}\n\n")
                    f.write(raw_response + "\n\n")
                    f.write("```json\n" + json.dumps(planner_usage, ensure_ascii=False, indent=2, default=str) + "\n```\n\n---\n\n")
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
                planner_response = raw_response
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
                logger.warning(f"Planner attempt {attempt} failed to produce valid JSON, retrying...")

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
                candidate_plans = [["Analyze the task instruction and implement the required changes"]]

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
                    logger.info(f"Plan {plan_idx}: estimated_dollar_cost={estimated_cost}, uncertainty={estimated_uncertainty}")
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
                        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens",
                                  "cache_read_tokens", "cache_write_tokens", "total_tokens"):
                            merged_ce[k] = merged_ce.get(k, 0) + m.get(k, 0)
                        merged_ce["accumulated_cost"] = merged_ce.get("accumulated_cost", 0.0) + m.get("accumulated_cost", 0.0)
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
        # Phase 2: Execution
        # ----------------
        # 创建执行代理（使用传入工具与 system_prompt_kwargs）
        executor_agent = Agent(
            llm=self.executor_llm,
            tools=self.tools,
            system_prompt_kwargs=self.system_prompt_kwargs,
        )

        step_index = 0
        response_id_to_step: dict[str, int] = {}
        execution_steps: list[ReactStepRecord] = []
        steps_json: list[dict[str, Any]] = []

        execution_prompt = render_execution_prompt(
            instruction=instruction,
            repo_path=repo_path,
            serialized_plan=serialized_plan,
        )
        try:
            with open(execution_traj, "w", encoding="utf-8") as f:
                f.write("# Execution Trajectory\n\n")
                if execution_meta:
                    f.write("## Agent Params (Execution)\n\n")
                    f.write("```json\n" + json.dumps(execution_meta, ensure_ascii=False, indent=2) + "\n```\n\n")
                f.write("## Initial Prompt\n\n")
                f.write("```\n" + execution_prompt + "\n```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init execution trajectory file {execution_traj}: {e}")

        def exec_callback(event):
            nonlocal step_index
            timestamp = time.time()
            role = getattr(event, "source", "environment")
            content_preview = CodeAgentPlanMode._event_preview(event)
            rec = ReactStepRecord(
                index=step_index,
                role=str(role),
                content_preview=content_preview[:500],
                metrics={},
                timestamp=timestamp,
                event_type=event.__class__.__name__,
            )
            # 记录 llm_response_id → step
            llm_resp_id = getattr(event, "llm_response_id", None)
            if llm_resp_id:
                response_id_to_step[str(llm_resp_id)] = step_index
            execution_steps.append(rec)

            # 逐步追加执行轨迹（模仿 SWE Runner）
            try:
                event_type = event.__class__.__name__
                step_payload: dict[str, Any] = {
                    "index": step_index,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "role": role,
                }

                # 为 ActionEvent 抽取 thought/action 与该响应对应的 token 使用情况
                if isinstance(event, ActionEvent):
                    # thought
                    thought_text = ""
                    thought = getattr(event, "thought", None)
                    if thought:
                        if isinstance(thought, list):
                            try:
                                thought_text = "".join(getattr(t, "text", str(t)) for t in thought)
                            except Exception:
                                thought_text = str(thought)
                        else:
                            thought_text = str(thought)

                    # action/tool
                    tool_name = getattr(event, "tool_name", "")
                    raw_args = getattr(getattr(event, "tool_call", None), "arguments", None)
                    parsed_args: Any = None
                    if raw_args is not None:
                        if isinstance(raw_args, (dict, list)):
                            parsed_args = raw_args
                        else:
                            s = str(raw_args)
                            try:
                                parsed_args = json.loads(s)
                            except Exception:
                                parsed_args = s

                    step_payload.update(
                        {
                            "thought": thought_text,
                            "action": {
                                "tool": tool_name,
                                "args": parsed_args,
                                "summary": getattr(event, "summary", None),
                            },
                            "llm_response_id": llm_resp_id,
                        }
                    )

                    # 从执行 LLM 的 metrics 中抓取与该 response_id 匹配的 usage/latency
                    step_metrics: dict[str, Any] = {}
                    try:
                        m = self.executor_llm.metrics.get()
                        # token usage per call
                        for u in m.get("token_usages", []) or []:
                            if str(u.get("response_id", "")) == str(llm_resp_id):
                                step_metrics.update(
                                    {
                                        "prompt_tokens": u.get("prompt_tokens", 0),
                                        "completion_tokens": u.get("completion_tokens", 0),
                                        "cache_read_tokens": u.get("cache_read_tokens", 0),
                                        "cache_write_tokens": u.get("cache_write_tokens", 0),
                                        "reasoning_tokens": u.get("reasoning_tokens", 0),
                                        "context_window": u.get("context_window", 0),
                                        "response_id": u.get("response_id", ""),
                                    }
                                )
                                # 新增：answer_tokens 等同于 completion_tokens（模型对外可见回答），与 reasoning_tokens 分开记录
                                try:
                                    step_metrics["answer_tokens"] = int(step_metrics.get("completion_tokens", 0) or 0)
                                except Exception:
                                    step_metrics["answer_tokens"] = step_metrics.get("completion_tokens", 0)
                                break
                        # response latency per call
                        for lat in m.get("response_latencies", []) or []:
                            if str(lat.get("response_id", "")) == str(llm_resp_id):
                                step_metrics["response_latency_sec"] = lat.get("latency", 0.0)
                                break
                    except Exception:
                        pass

                    rec.metrics = step_metrics
                    step_payload["metrics"] = step_metrics

                # 为 ObservationEvent 抽取 observation 输出
                elif isinstance(event, ObservationEvent):
                    obs = getattr(event, "observation", None)
                    content = ""
                    if obs is not None:
                        try:
                            content = "".join(content_to_str(obs.to_llm_content))
                        except Exception:
                            content = str(obs)
                    step_payload["observation"] = content

                # 持久化到 trajectory.md
                with open(execution_traj, "a", encoding="utf-8") as f:
                    f.write(f"## {event_type}\n")
                    if isinstance(event, ActionEvent):
                        thought = getattr(event, "thought", None)
                        if thought:
                            if isinstance(thought, list):
                                text = "".join(getattr(t, "text", str(t)) for t in thought)
                            else:
                                text = str(thought)
                            f.write(f"**Thought**: {text}\n\n")
                        f.write(f"**Tool**: {getattr(event, 'tool_name', '')}\n")
                        args = getattr(getattr(event, "tool_call", None), "arguments", None)
                        if args:
                            f.write(f"**Args**: ```json\n{args}\n```\n\n")
                        # 写入当前 step 的 metrics（若有）
                        if rec.metrics:
                            try:
                                f.write("**Metrics**: ```json\n" + json.dumps(rec.metrics, ensure_ascii=False, indent=2) + "\n```\n\n")
                            except Exception:
                                pass
                    elif isinstance(event, ObservationEvent):
                        obs = getattr(event, "observation", None)
                        content = ""
                        if obs is not None:
                            content = "".join(content_to_str(obs.to_llm_content))
                        f.write(f"**Observation**: ```\n{content}\n```\n\n")
                    f.write("---\n\n")

                # 收集结构化步骤信息（JSON）
                steps_json.append(step_payload)
            except Exception as ex:
                logger.warning(f"Error saving execution trajectory: {ex}")

            step_index += 1

        exec_conv = Conversation(
            agent=executor_agent,
            workspace=workspace,
            callbacks=(callbacks or []) + [exec_callback],
        )

        exec_conv.send_message(execution_prompt)
        run_conversation_with_fake_user_response(exec_conv)
        exec_metrics = self._conversation_metrics(exec_conv)

        # 写出结构化的步骤日志（JSON）：初始 prompt + 每一步输出与该步 metrics
        try:
            self._write_json(
                os.path.join(logs_dir, "execution_steps.json"),
                {"initial_prompt": execution_prompt, "steps": steps_json},
            )
        except Exception as e:
            logger.warning(f"Failed to write execution_steps.json: {e}")

        # 将 token_usages 映射到步骤（通过 response_id）
        per_step: list[dict[str, Any]] = []
        for usage in exec_metrics.get("token_usages", []):
            resp_id = usage.get("response_id") or ""
            step_id = response_id_to_step.get(str(resp_id), None)
            item = {
                "response_id": resp_id,
                "step_index": step_id,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
                "cache_write_tokens": usage.get("cache_write_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            }
            # 新增：answer_tokens（等同 completion_tokens，用于区分与 reasoning_tokens 的合计关系）
            try:
                item["answer_tokens"] = int(item.get("completion_tokens", 0) or 0)
            except Exception:
                item["answer_tokens"] = item.get("completion_tokens", 0)
            per_step.append(item)

        # 采用逐步追加策略，不再生成覆盖式全量轨迹快照

        # ----------------
        # 汇总与产物输出
        # ----------------
        total_metrics = self._merge_metrics_dict(planner_metrics, exec_metrics)
        token_report: dict[str, Any] = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "execution_per_step": per_step,
            "total": total_metrics,
        }
        if ce_metrics:
            token_report["cost_estimation"] = ce_metrics

        self._write_json(os.path.join(logs_dir, "token_usage.json"), token_report)

        def _summarize_usage(m: dict[str, Any]) -> dict[str, int]:
            u = (m.get("accumulated_token_usage") or {})
            if u:
                comp = int(u.get("completion_tokens", 0) or 0)
                reas = int(u.get("reasoning_tokens", 0) or 0)
                return {
                    "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                    "completion_tokens": comp,
                    "answer_tokens": comp,
                    "cache_read_tokens": int(u.get("cache_read_tokens", 0) or 0),
                    "cache_write_tokens": int(u.get("cache_write_tokens", 0) or 0),
                    "reasoning_tokens": reas,
                }
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
            "execution_per_step": per_step,
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
        log_lines.extend([
            f"- Planner tokens: {_summarize_usage(planner_metrics)}",
            f"- Executor tokens: {_summarize_usage(exec_metrics)}",
            f"- Total tokens: {_summarize_usage(total_metrics)}",
        ])
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
            conversation=exec_conv,
            execution_trajectory=execution_steps,
            execution_steps_metrics=per_step,
            other_content=other,
        )

        return result
