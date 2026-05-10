"""
TraeCodeAgentPlanMode — Trae Agent baseline 规划-执行双阶段。

阶段：
  1) 工作区扫描：在容器内扫描工作区，生成目录结构描述
  2) 规划：使用 planner LLM 读取任务指令 + 工作区概览，生成 JSON 格式的实现计划
  3) 执行：将计划序列化后注入 prompt，在容器内运行 trae-cli 执行

与自定义 CodeAgentPlanMode 的区别：
  - 规划阶段使用 OpenHands 风格的 planning prompt（baseline）
  - 执行阶段使用 Trae Agent 原生的 system prompt 和工具链
  - 作为 baseline 与自定义优化版对比
"""

from __future__ import annotations

import json
import os
import re
import shlex
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2
from src.module.gpt_inference import SimpleAPICaller
from src.agent.trae_code_agent import (
    TraeCodeAgent,
    AgentResult,
    TraeArtifactBuilder,
    generate_trae_config_yaml,
    _put_file_in_container,
)

logger = logging.getLogger(__name__)


def render_trae_planning_prompt(
    instruction: str,
    workspace_overview: str,
    base_dir: Optional[str] = None,
) -> str:
    return render_j2(
        "trae_baseline_planning.j2",
        {"instruction": instruction, "workspace_overview": workspace_overview},
        base_dir=base_dir,
    )


def render_trae_execution_prompt(
    instruction: str, repo_path: str, serialized_plan: str, base_dir: Optional[str] = None
) -> str:
    return render_j2(
        "trae_baseline_execution.j2",
        {"instruction": instruction, "repo_path": repo_path, "serialized_plan": serialized_plan},
        base_dir=base_dir,
    )


class TraeCodeAgentPlanMode:
    """Trae Agent baseline 规划-执行双阶段。

    规划阶段：使用 SimpleAPICaller 调用 planner LLM
    执行阶段：在容器内运行 trae-cli
    """

    def __init__(
        self,
        planner_cfg: dict | None = None,
        executor_llm: Any | None = None,
        artifacts_dir: str | Path | None = None,
        planner_llm: Any | None = None,
    ):
        if planner_cfg is None and planner_llm is not None:
            self.planner_caller = SimpleAPICaller(
                llm_name=getattr(planner_llm, "model", "").replace("openai/", ""),
                api_key=getattr(planner_llm, "api_key", ""),
                base_url=getattr(planner_llm, "base_url", None),
                api_version=getattr(planner_llm, "api_version", None),
                cache_server_url=getattr(planner_llm, "cache_server_url", None),
            )
        else:
            self.planner_caller = SimpleAPICaller(
                llm_name=planner_cfg["llm_name"],
                api_key=planner_cfg["key"],
                base_url=planner_cfg.get("openai_base_url"),
                api_version=planner_cfg.get("api_version"),
                cache_server_url=planner_cfg.get("cache_server_url"),
            )
        self.executor_llm = executor_llm
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    @staticmethod
    def _merge_metrics_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
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
    'data', 'dataset', 'datasets', 'fixtures',
    'examples', 'example',
}

SOURCE_EXTS = {
    '.py', '.pyx', '.pxd',
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '.java', '.kt', '.scala', '.groovy',
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
    '.go', '.rs', '.swift', '.m', '.mm',
    '.rb', '.pl', '.pm', '.lua', '.r', '.R',
    '.sh', '.bash', '.zsh', '.fish',
    '.sql',
}

CONFIG_EXTS = {
    '.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf',
    '.xml', '.properties', '.env', '.dotenv',
    '.cmake', '.make', '.mk', '.gradle',
    '.dockerfile',
}

DOC_EXTS = {
    '.md', '.rst', '.txt', '.adoc', '.org',
    '.html', '.css', '.scss', '.less',
}

def is_low_priority(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    return any(p in LOW_PRIORITY_DIRS for p in parts)

def relevance_score(entry):
    score = 0
    path = entry['path']
    ext = os.path.splitext(path)[1].lower()
    if entry['binary']:
        return -100
    if entry['low_priority']:
        score -= 50
    if ext in SOURCE_EXTS:
        score += 30
    elif ext in CONFIG_EXTS:
        score += 5
    elif ext in DOC_EXTS:
        score -= 10
    else:
        score -= 20
    depth = path.replace('\\', '/').count('/')
    if depth <= 1:
        score += 15
    elif depth <= 3:
        score += 5
    basename = os.path.basename(path).lower()
    high_priority_names = {
        'main', 'app', 'index', '__init__', 'setup', 'conftest',
        'config', 'settings', 'manage', 'wsgi', 'asgi',
        'makefile', 'dockerfile', 'cargo', 'package',
    }
    name_no_ext = os.path.splitext(basename)[0]
    if name_no_ext in high_priority_names:
        score += 10
    if entry['lines'] > 0:
        if entry['lines'] <= 50:
            score += 5
        elif entry['lines'] <= 500:
            score += 3
        elif entry['lines'] > 2000:
            score -= 3
    return score

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

for e in all_entries:
    e['score'] = relevance_score(e)

non_binary = [e for e in all_entries if not e['binary']]
non_binary.sort(key=lambda e: e['score'], reverse=True)

high = [e for e in non_binary if not e['low_priority']]
low = [e for e in non_binary if e['low_priority']]

dir_counts = {}
for e in all_entries:
    top = e['path'].replace('\\', '/').split('/')[0]
    if top not in dir_counts:
        dir_counts[top] = {'total': 0, 'source': 0}
    dir_counts[top]['total'] += 1
    if not e['low_priority'] and not e['binary']:
        dir_counts[top]['source'] += 1

print(json.dumps({
    "high_priority": high,
    "low_priority": low,
    "total_on_disk": total_files_on_disk,
    "dir_counts": dir_counts,
    "non_binary_count": len(non_binary),
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
            non_binary_count = scan_data.get("non_binary_count", total_on_disk)
            dir_counts = scan_data.get("dir_counts", {})
        except Exception as e:
            logger.warning(f"Workspace scan error: {e}")
            return f"(workspace scan error: {e}, working directory: {repo_path})"

        out: list[str] = [f"Workspace root: {repo_path}"]
        out.append(f"Total files: {total_on_disk} ({non_binary_count} text files)")

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

        high_budget = max_files if len(high_priority) <= max_files else int(max_files * 0.85)
        low_budget = max_files_per_dir if len(high_priority) <= max_files else max(0, max_files - high_budget)

        out.append("")
        out.append("Source files (sorted by relevance):")
        out.append(f"{'Path':<62} {'Lines':>6}")
        out.append("-" * 70)

        shown = 0
        for e in high_priority:
            if shown >= high_budget:
                remaining = len(high_priority) - shown
                if remaining > 0:
                    out.append(f"  ... ({remaining} more source files omitted)")
                break
            lines_str = str(e["lines"]) if e["lines"] >= 0 else "-"
            path_display = e["path"]
            if len(path_display) > 60:
                path_display = "..." + path_display[-57:]
            out.append(f"{path_display:<62} {lines_str:>6}")
            shown += 1

        if low_priority and low_budget > 0:
            out.append("")
            out.append(f"Other files ({len(low_priority)} total, showing up to {low_budget}):")
            out.append(f"{'Path':<62} {'Lines':>6}")
            out.append("-" * 70)
            shown_lp = 0
            for e in low_priority:
                if shown_lp >= low_budget:
                    out.append(f"  ... ({len(low_priority) - shown_lp} more omitted)")
                    break
                lines_str = str(e["lines"]) if e["lines"] >= 0 else "-"
                path_display = e["path"]
                if len(path_display) > 60:
                    path_display = "..." + path_display[-57:]
                out.append(f"{path_display:<62} {lines_str:>6}")
                shown_lp += 1
        elif low_priority and low_budget == 0:
            out.append("")
            out.append(f"({len(low_priority)} non-source files omitted to prioritize relevant code)")

        return "\n".join(out)

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

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_trae_plan")
        logs_dir = os.path.join(out_dir, "log")
        self._ensure_dirs(logs_dir)

        logger.info(f"Scanning workspace at {repo_path} (max {max_scan_files} files)...")
        workspace_overview = self.scan_workspace(
            workspace=workspace,
            repo_path=repo_path,
            max_files=max_scan_files,
        )
        logger.info(f"Workspace scan complete, overview length: {len(workspace_overview)} chars")
        self._write_text(os.path.join(logs_dir, "workspace_overview.txt"), workspace_overview)

        planning_prompt = render_trae_planning_prompt(
            instruction=instruction,
            workspace_overview=workspace_overview,
        )

        try:
            self._write_text(os.path.join(logs_dir, "planner_prompt.txt"), planning_prompt)
        except Exception as e:
            logger.warning(f"Failed to save planner prompt: {e}")

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
                self._write_text(
                    os.path.join(logs_dir, f"planner_attempt_{attempt}.txt"),
                    raw_response,
                )
            except Exception as e:
                logger.warning(f"Failed to save planner attempt: {e}")

            single = self._extract_plans_from_text(raw_response, multi=False)
            parsed_plans = [single] if single else None

            if parsed_plans is not None:
                break

            if attempt < max_plan_attempts:
                retry_msg = (
                    "Your previous response could not be parsed as valid JSON. "
                    "Please output the plan again as a valid JSON code block (```json ... ```).\n"
                    "Expected format: a JSON array of strings (each string is one subtask).\n"
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

        json_plan = parsed_plans[0] if parsed_plans else None

        if json_plan is None:
            logger.warning("Failed to extract JSON plan after retries, using fallback")
            json_plan = ["Analyze the task instruction and implement the required changes"]

        self._write_json(os.path.join(logs_dir, "plan.json"), json_plan)
        serialized_plan = self._serialize_plan_for_execution(json_plan)
        self._write_text(os.path.join(logs_dir, "plan_serialized.md"), serialized_plan)
        logger.info(f"Plan extracted: {len(json_plan)} subtasks")

        if self.executor_llm is not None:
            executor_cfg = {
                "llm_name": getattr(self.executor_llm, "model", "").replace("openai/", ""),
                "key": getattr(self.executor_llm, "api_key", ""),
                "openai_base_url": getattr(self.executor_llm, "base_url", None),
                "api_version": getattr(self.executor_llm, "api_version", None),
            }
        else:
            executor_cfg = {
                "llm_name": self.planner_caller.llm_name,
                "key": self.planner_caller.api_key,
                "openai_base_url": self.planner_caller.base_url,
                "api_version": self.planner_caller.api_version,
            }

        artifacts_dir = self.artifacts_dir or Path(out_dir)

        executor_agent = TraeCodeAgent(
            llm_cfg=executor_cfg,
            artifacts_dir=artifacts_dir,
            max_steps=100,
            repo_path=repo_path,
        )

        execution_prompt = render_trae_execution_prompt(
            instruction=instruction,
            repo_path=repo_path,
            serialized_plan=serialized_plan,
        )

        try:
            self._write_text(os.path.join(logs_dir, "execution_prompt.txt"), execution_prompt)
        except Exception as e:
            logger.warning(f"Failed to save execution prompt: {e}")

        logger.info("Running TraeCodeAgent executor...")
        exec_result = executor_agent.run(
            instruction=execution_prompt,
            workspace=workspace,
            output_dir=logs_dir,
        )
        exec_metrics = exec_result.metrics

        total_metrics = self._merge_metrics_dict(planner_metrics, exec_metrics)
        token_report: dict[str, Any] = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "total": total_metrics,
        }

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
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        all_metrics: dict[str, Any] = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "total": total_metrics,
        }

        other: dict[str, Any] = {
            "logs_dir": logs_dir,
            "plan_json": json_plan,
            "git_patch": exec_result.other_content.get("git_patch", ""),
        }

        result = AgentResult(
            metrics=all_metrics,
            conversation=None,
        )
        result.other_content = other

        return result
