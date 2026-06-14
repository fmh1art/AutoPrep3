"""
并行运行SWE-bench评估脚本 — CustomizedCodeAgent + 新的 PyCodeGraph

用法示例：
    python example/benchmark_code_agent_with_pycodegraph.py \
      --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
      --split test \
      --eval-limit 64 \
      --exp-config _config/doubao.yaml \
      --max-steps 200 \
      --parallel 8 \
      --use-pycodegraph \
      --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
      --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME
from src.benchmarks.swebench.swe_eval import run_swebench_eval
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.funcs import render_j2
from src.agent.code_agent import CustomizedCodeAgent
from src.tools.codegraph_setup import (
    setup_codegraph,
    setup_codegraph_py,
    setup_pycodegraph,
)

logger = logging.getLogger(__name__)

for _ln in ("uvicorn.access", "uvicorn.error", "httpcore", "httpx"):
    logging.getLogger(_ln).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]

# 三选一：优先级 PyCodeGraph > CodeGraphPy > CodeGraph
CODE_GRAPH_KIND_NONE = "none"
CODE_GRAPH_KIND_OLD = "codegraph"
CODE_GRAPH_KIND_NEW_PY = "codegraph_py"
CODE_GRAPH_KIND_PYCODEGRAPH = "pycodegraph"


def _resolve_code_graph_kind(args) -> str:
    """根据 CLI 参数解析出最终启用哪种 code graph。"""
    if getattr(args, "use_pycodegraph", False):
        return CODE_GRAPH_KIND_PYCODEGRAPH
    if getattr(args, "use_new_code_graph", False):
        return CODE_GRAPH_KIND_NEW_PY
    if getattr(args, "use_codegraph", False):
        return CODE_GRAPH_KIND_OLD
    return CODE_GRAPH_KIND_NONE


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _resolve_path(value: str, fallback_base: Path | None = None) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    if fallback_base is not None:
        return str((fallback_base / p).resolve())
    return str((REPO_ROOT / p).resolve())


def _log_eval_progress(result: dict, results_so_far: list[dict], total_instances: int) -> None:
    instance_id = result.get("instance_id", "?")
    has_error = bool(result.get("error"))
    resolved = bool(result.get("resolved", False))

    done = len(results_so_far)
    resolved_count = sum(1 for r in results_so_far if r.get("resolved", False))
    error_count = sum(1 for r in results_so_far if r.get("error"))
    rate = (resolved_count / done * 100) if done > 0 else 0.0

    status = "ERROR " if has_error else ("RESOLVED=True " if resolved else "RESOLVED=False")
    parts = [
        f"[Eval {done:>3}/{total_instances}] {instance_id} → {status}",
        f"running {resolved_count}/{done} ({rate:.1f}%)",
        f"errors={error_count}",
    ]
    if has_error:
        err_preview = str(result.get("error", ""))[:120]
        parts.append(f"err={err_preview!r}")
    logger.info(" | ".join(parts))


def _snapshot_initial_workspace(
    workspace,
    repo_path: str,
    snapshot_root: str,
    instance_id: str,
    repo_name: str,
) -> str | None:
    if not snapshot_root:
        return None

    container_id = getattr(workspace, "_container_id", None)
    if not container_id:
        logger.warning(f"[Worker] Cannot snapshot {instance_id}: workspace has no container id")
        return None

    safe_instance_id = _safe_path_part(instance_id)
    safe_repo_name = _safe_path_part(repo_name)
    snapshot_dir = Path(snapshot_root) / safe_instance_id / safe_repo_name

    try:
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["docker", "cp", f"{container_id}:{repo_path}/.", str(snapshot_dir)],
            capture_output=True,
            text=True,
            timeout=float(os.getenv("INITIAL_WORKSPACE_SNAPSHOT_TIMEOUT", "300")),
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"[Worker] Failed to snapshot initial workspace for {instance_id}: "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )
            return None

        logger.info(
            f"[Worker] Snapshotted initial workspace for {instance_id} to {snapshot_dir}"
        )
        return str(snapshot_dir)
    except Exception as e:
        logger.warning(f"[Worker] Error snapshotting initial workspace for {instance_id}: {e}")
        return None


def _check_python_ratio(workspace, repo_path: str, timeout: float = 30.0) -> float:
    """
    检查项目中Python代码的比例。

    通过统计文件扩展名来计算Python代码占比。
    只统计源代码文件，忽略构建产物、依赖目录等。

    Returns:
        Python文件占源代码文件总数的比例 (0.0 - 1.0)
    """
    CODE_EXTENSIONS = {
        ".py", ".pyi",
        ".js", ".jsx", ".ts", ".tsx",
        ".java", ".kt", ".scala",
        ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs",
        ".rb", ".php",
        ".cs",
        ".swift", ".m", ".mm",
        ".sh", ".bash",
        ".lua", ".pl",
        ".r",
        ".ex", ".exs",
        ".clj", ".cljs",
        ".hs",
        ".ml", ".mli",
        ".erl", ".hrl",
    }
    IGNORE_DIRS = {
        ".git", ".svn", ".hg",
        "node_modules",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "venv", ".venv", "env",
        "build", "dist",
        "target",
        ".next", ".nuxt",
        "vendor",
        "third_party",
        "third-party",
        "lib", "libs",
        "deps",
    }

    try:
        find_cmd = (
            f"cd {repo_path} && "
            f"find . -type f "
            f"-not -path '*/\\.*' "
        )
        for ignore_dir in IGNORE_DIRS:
            find_cmd += f"-not -path '*/{ignore_dir}/*' "

        ext_pattern = '|'.join('\\.' + ext.lstrip('.') for ext in CODE_EXTENSIONS)
        total_cmd = find_cmd + f"| grep -E '\\.({ext_pattern})$' | wc -l"
        py_cmd = find_cmd + "| grep -E '\\.(py|pyi)$' | wc -l"

        total_result = workspace.execute_command(total_cmd, timeout=timeout)
        py_result = workspace.execute_command(py_cmd, timeout=timeout)

        if total_result.exit_code != 0 or py_result.exit_code != 0:
            logger.warning(
                f"[Worker] Failed to count files: "
                f"total exit={total_result.exit_code}, py exit={py_result.exit_code}"
            )
            return 0.0

        try:
            total_files = int(total_result.stdout.strip())
            py_files = int(py_result.stdout.strip())
        except (ValueError, TypeError):
            logger.warning(
                f"[Worker] Failed to parse file counts: "
                f"total='{total_result.stdout}', py='{py_result.stdout}'"
            )
            return 0.0

        if total_files == 0:
            logger.warning(f"[Worker] No source code files found in {repo_path}")
            return 0.0

        ratio = py_files / total_files
        logger.info(
            f"[Worker] File count: {py_files} Python / {total_files} total source files "
            f"({ratio:.1%})"
        )
        return ratio

    except Exception as e:
        logger.warning(f"[Worker] Error checking Python ratio: {e}")
        return 0.0


def run_single_instance(args_dict):
    """Worker: 对单个 SWE-bench 实例执行完整的 clone → (可选)构建 PyCodeGraph → agent.run → git patch → eval 流程。"""
    instance = args_dict["instance"]
    llm_cfg = args_dict["llm_cfg"]
    max_steps = args_dict["max_steps"]
    tmp_root = args_dict["tmp_root"]
    runner_config = args_dict["runner_config"]
    agent_type = args_dict.get("agent_type", "code")
    max_steps_per_subagent = args_dict.get("max_steps_per_subagent", 80)
    skill_path = args_dict.get("skill_path")
    code_graph_kind = args_dict.get("code_graph_kind", CODE_GRAPH_KIND_NONE)
    initial_workspaces_dir = args_dict.get("initial_workspaces_dir")

    instance_id = instance["instance_id"]
    repo_name = instance["repo"].split("/")[-1]
    repo_path = f"/workspace/{repo_name}"
    base_commit = instance["base_commit"]
    repo_url = f"https://github.com/{instance['repo']}.git"

    log_dir = os.path.join(tmp_root, "log", instance_id)
    os.makedirs(log_dir, exist_ok=True)

    # 根据项目语言比例决定是否启用 code graph（Python 项目才启用）
    use_codegraph_for_instance = (code_graph_kind == CODE_GRAPH_KIND_OLD)
    use_new_code_graph_for_instance = (code_graph_kind == CODE_GRAPH_KIND_NEW_PY)
    use_pycodegraph_for_instance = (code_graph_kind == CODE_GRAPH_KIND_PYCODEGRAPH)

    workspace = None
    try:
        runner = SweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)

        repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
        clone_result = workspace.execute_command(
            f"rm -rf {repo_path} && "
            f"git init {repo_path} && "
            f"cd {repo_path} && "
            f"git remote add origin {repo_url} && "
            f"git fetch --depth 1 origin {base_commit}",
            timeout=float(repo_prepare_timeout),
        )
        if clone_result.exit_code != 0:
            raise RuntimeError(
                f"git fetch failed for {instance_id}: {clone_result.stderr or clone_result.stdout}"
            )

        checkout_result = workspace.execute_command(
            f"cd {repo_path} && git checkout --detach FETCH_HEAD",
            timeout=120.0,
        )
        if checkout_result.exit_code != 0:
            raise RuntimeError(
                f"git checkout failed for {instance_id}: {checkout_result.stderr or checkout_result.stdout}"
            )

        _snapshot_initial_workspace(
            workspace=workspace,
            repo_path=repo_path,
            snapshot_root=initial_workspaces_dir,
            instance_id=instance_id,
            repo_name=repo_name,
        )

        # 对启用 code graph 的项目进行 Python 比例检测（本项目专为 Python 项目优化，
        # 对非 Python 项目降级为普通 agent）
        any_code_graph_enabled = (
            use_codegraph_for_instance
            or use_new_code_graph_for_instance
            or use_pycodegraph_for_instance
        )
        if any_code_graph_enabled:
            python_ratio = _check_python_ratio(workspace, repo_path)
            logger.info(
                f"[Worker] Python code ratio for {instance_id}: {python_ratio:.1%}"
            )
            PYTHON_RATIO_THRESHOLD = 0.0
            if python_ratio < PYTHON_RATIO_THRESHOLD:
                logger.warning(
                    f"[Worker] Python code ratio ({python_ratio:.1%}) below threshold "
                    f"({PYTHON_RATIO_THRESHOLD:.0%}) for {instance_id}; "
                    f"disabling code graph, falling back to normal code agent."
                )
                use_codegraph_for_instance = False
                use_new_code_graph_for_instance = False
                use_pycodegraph_for_instance = False

        # 按优先级安装并构建对应的 code graph
        if use_pycodegraph_for_instance:
            # 新的 PyCodeGraph
            logger.info(f"[Worker] Setting up PyCodeGraph for {instance_id} ...")
            ok = setup_pycodegraph(workspace, repo_path)
            if not ok:
                logger.warning(
                    f"[Worker] PyCodeGraph setup failed for {instance_id}; "
                    f"agent will fall back to grep/Read."
                )
                use_pycodegraph_for_instance = False
        elif use_new_code_graph_for_instance:
            logger.info(f"[Worker] Setting up CodeGraphPy (new) for {instance_id} ...")
            ok = setup_codegraph_py(workspace, repo_path)
            if not ok:
                logger.warning(
                    f"[Worker] CodeGraphPy setup failed for {instance_id}; "
                    f"agent will fall back to grep/Read."
                )
                use_new_code_graph_for_instance = False
        elif use_codegraph_for_instance:
            logger.info(f"[Worker] Setting up original CodeGraph for {instance_id} ...")
            ok = setup_codegraph(workspace, repo_path)
            if not ok:
                logger.warning(
                    f"[Worker] CodeGraph setup failed for {instance_id}; "
                    f"agent will fall back to grep/Read."
                )
                use_codegraph_for_instance = False

        task_description = render_j2(
            template_name="query.j2",
            context={
                "repo_path": repo_path,
                "problem_statement": str(instance.get("problem_statement", "")).strip(),
                "base_commit": base_commit,
            },
        )

        if agent_type == "meta":
            pass
        else:
            agent = CustomizedCodeAgent(
                llm_cfg=llm_cfg,
                output_dir=log_dir,
                max_step=max_steps,
                skill_path=skill_path,
                use_codegraph=use_codegraph_for_instance,
                use_new_code_graph=use_new_code_graph_for_instance,
                use_pycodegraph=use_pycodegraph_for_instance,
            )
            agent.run(
                task_instruction=task_description,
                workspace=workspace,
            )

        workspace.execute_command(
            f"cd {repo_path} && "
            f"find . -name '*.bak' -delete && "
            f"find . -name '*.orig' -delete && "
            f"rm -f reproduce_issue.py test_bug.py test_simple.py test_fix.py"
        )
        workspace.execute_command(f"cd {repo_path} && git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global user.email '{GIT_USER_EMAIL}' && "
            f"git config --global user.name '{GIT_USER_NAME}' && "
            f"git commit --no-verify -m '{GIT_COMMIT_MESSAGE}' || true"
        )

        diff_result = workspace.execute_command(
            f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD"
        )
        git_patch = diff_result.stdout if diff_result.exit_code == 0 else ""

        eval_result = run_swebench_eval(
            instance=instance,
            git_patch=git_patch,
            run_id=str(uuid.uuid4())[:8],
            tmp_dir=tmp_root,
        )

        result = {
            "instance_id": instance_id,
            "git_patch": git_patch,
            "resolved": eval_result.get("resolved", False),
            "patch_applied": eval_result.get("patch_applied", False),
        }
        logger.info(f"[Worker] Completed {instance_id}: resolved={result['resolved']}")
        return result

    except Exception as e:
        logger.error(f"[Worker] Error in {instance_id}: {e}")
        return {
            "instance_id": instance_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "resolved": False,
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Cleanup failed for {instance_id}: {cleanup_err}")


def _load_pricing(llm_cfg: dict) -> dict:
    pricing = {
        "currency": "unknown",
        "input_token_price": 0.0,
        "output_token_price": 0.0,
        "cached_token_price": 0.0,
    }

    if "price_yuan_per_million_token" in llm_cfg:
        p = llm_cfg["price_yuan_per_million_token"]
        pricing["currency"] = "yuan"
        pricing["input_token_price"] = float(p.get("input_token", 0)) / 1_000_000
        pricing["output_token_price"] = float(p.get("output_token", 0)) / 1_000_000
        pricing["cached_token_price"] = float(p.get("cached_token", 0)) / 1_000_000
        pricing["price_yuan_per_million_token"] = {
            "input_token": float(p.get("input_token", 0)),
            "output_token": float(p.get("output_token", 0)),
            "cached_token": float(p.get("cached_token", 0)),
        }
    elif "price_dollar_per_token" in llm_cfg:
        p = llm_cfg["price_dollar_per_token"]
        pricing["currency"] = "dollar"
        pricing["input_token_price"] = float(p.get("input_token", 0))
        pricing["output_token_price"] = float(p.get("output_token", 0))
        pricing["cached_token_price"] = float(p.get("cached_token", 0))
        pricing["price_dollar_per_token"] = {
            "input_token": float(p.get("input_token", 0)),
            "output_token": float(p.get("output_token", 0)),
            "cached_token": float(p.get("cached_token", 0)),
        }

    return pricing


def _calculate_cost(uncached_input_tokens: int, output_tokens: int, cached_input_tokens: int, pricing: dict) -> float:
    cost = (
        uncached_input_tokens * pricing["input_token_price"]
        + output_tokens * pricing["output_token_price"]
        + cached_input_tokens * pricing["cached_token_price"]
    )
    return cost


def generate_total_records(output_dir: str, llm_cfg: dict, exp_config_name: str) -> dict:
    results_path = os.path.join(output_dir, "results.json")
    if not os.path.isfile(results_path):
        logger.warning(f"[TotalRecords] results.json not found at {results_path}, skipping.")
        return {}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    pricing = _load_pricing(llm_cfg)
    currency = pricing["currency"]
    cost_suffix = f"cost_{currency}"

    per_instance = []
    for result in results:
        instance_id = result.get("instance_id", "?")
        resolved = bool(result.get("resolved", False))
        has_error = bool(result.get("error"))

        records_path = os.path.join(output_dir, "log", instance_id, "records.json")
        if not os.path.isfile(records_path):
            logger.warning(f"[TotalRecords] records.json not found for {instance_id}")
            per_instance.append({
                "instance_id": instance_id,
                "resolved": resolved,
                "has_error": has_error,
                "records_found": False,
            })
            continue

        try:
            with open(records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            logger.warning(f"[TotalRecords] Failed to load records for {instance_id}: {e}")
            per_instance.append({
                "instance_id": instance_id,
                "resolved": resolved,
                "has_error": has_error,
                "records_found": False,
            })
            continue

        overall = records.get("overall", {})
        total_steps = overall.get("total_steps", 0)
        input_tokens = overall.get("input_tokens", 0)
        output_tokens = overall.get("output_tokens", 0)
        cached_input_tokens = overall.get("cached_tokens", 0)
        uncached_input_tokens = overall.get("uncached_tokens", 0)
        reasoning_tokens = overall.get("reasoning_tokens", 0)
        total_tokens = overall.get("total_tokens", 0)
        use_time = overall.get("use_time", 0)

        execution = overall.get("execution", {})
        exec_input_tokens = execution.get("input_tokens", 0)
        exec_output_tokens = execution.get("output_tokens", 0)
        exec_cached_input_tokens = execution.get("cached_tokens", 0)
        exec_uncached_input_tokens = execution.get("uncached_tokens", 0)
        exec_reasoning_tokens = execution.get("reasoning_tokens", 0)
        exec_total_tokens = execution.get("total_tokens", 0)
        exec_use_time = execution.get("use_time", 0)

        combined_input = input_tokens + exec_input_tokens
        combined_output = output_tokens + exec_output_tokens
        combined_cached_input = cached_input_tokens + exec_cached_input_tokens
        combined_uncached_input = uncached_input_tokens + exec_uncached_input_tokens
        combined_reasoning = reasoning_tokens + exec_reasoning_tokens
        combined_total = total_tokens + exec_total_tokens
        combined_use_time = use_time + exec_use_time

        cost = _calculate_cost(combined_uncached_input, combined_output, combined_cached_input, pricing)

        inst_record = {
            "instance_id": instance_id,
            "resolved": resolved,
            "has_error": has_error,
            "records_found": True,
            "total_steps": total_steps,
            "input_tokens": combined_input,
            "output_tokens": combined_output,
            "cached_input_tokens": combined_cached_input,
            "uncached_input_tokens": combined_uncached_input,
            "reasoning_tokens": combined_reasoning,
            "total_tokens": combined_total,
            "use_time_seconds": round(combined_use_time, 2),
            cost_suffix: round(cost, 6),
        }

        if execution:
            inst_record["meta_agent_tokens"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "uncached_input_tokens": uncached_input_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "use_time": round(use_time, 2),
            }
            inst_record["execution_tokens"] = {
                "input_tokens": exec_input_tokens,
                "output_tokens": exec_output_tokens,
                "cached_input_tokens": exec_cached_input_tokens,
                "uncached_input_tokens": exec_uncached_input_tokens,
                "reasoning_tokens": exec_reasoning_tokens,
                "total_tokens": exec_total_tokens,
                "use_time": round(exec_use_time, 2),
            }

        per_instance.append(inst_record)

    valid_instances = [inst for inst in per_instance if inst.get("records_found", False)]
    resolved_instances = [inst for inst in valid_instances if inst.get("resolved", False)]
    error_instances = [inst for inst in per_instance if inst.get("has_error", False)]

    def _safe_sum(key):
        return sum(inst.get(key, 0) for inst in valid_instances)

    def _safe_median(key):
        vals = [inst.get(key, 0) for inst in valid_instances]
        return median(vals) if vals else 0.0

    total_valid = len(valid_instances)
    total_all = len(per_instance)

    overall_stats = {
        "total_instances": total_all,
        "valid_instances": total_valid,
        "resolved_count": len(resolved_instances),
        "resolved_rate": round(len(resolved_instances) / total_all, 4) if total_all > 0 else 0.0,
        "error_count": len(error_instances),
        "total_steps": _safe_sum("total_steps"),
        "avg_steps": round(_safe_sum("total_steps") / total_valid, 2) if total_valid > 0 else 0.0,
        "median_steps": round(_safe_median("total_steps"), 2),
        "total_input_tokens": _safe_sum("input_tokens"),
        "total_output_tokens": _safe_sum("output_tokens"),
        "total_cached_input_tokens": _safe_sum("cached_input_tokens"),
        "total_uncached_input_tokens": _safe_sum("uncached_input_tokens"),
        "total_reasoning_tokens": _safe_sum("reasoning_tokens"),
        "total_tokens": _safe_sum("total_tokens"),
        "avg_input_tokens": round(_safe_sum("input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_output_tokens": round(_safe_sum("output_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_cached_input_tokens": round(_safe_sum("cached_input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_uncached_input_tokens": round(_safe_sum("uncached_input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_total_tokens": round(_safe_sum("total_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "total_use_time_seconds": round(_safe_sum("use_time_seconds"), 2),
        "avg_use_time_seconds": round(_safe_sum("use_time_seconds") / total_valid, 2) if total_valid > 0 else 0.0,
        f"total_{cost_suffix}": round(_safe_sum(cost_suffix), 6),
        f"avg_{cost_suffix}": round(_safe_sum(cost_suffix) / total_valid, 6) if total_valid > 0 else 0.0,
        f"cost_per_resolved_{currency}": round(_safe_sum(cost_suffix) / len(resolved_instances), 6) if resolved_instances else 0.0,
    }

    if resolved_instances:
        overall_stats["resolved_avg_steps"] = round(
            sum(inst.get("total_steps", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_input_tokens"] = round(
            sum(inst.get("input_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_output_tokens"] = round(
            sum(inst.get("output_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_total_tokens"] = round(
            sum(inst.get("total_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats[f"resolved_avg_{cost_suffix}"] = round(
            sum(inst.get(cost_suffix, 0) for inst in resolved_instances) / len(resolved_instances), 6
        )
        overall_stats["resolved_avg_use_time_seconds"] = round(
            sum(inst.get("use_time_seconds", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )

    total_records = {
        "config": {
            "exp_config": exp_config_name,
            "llm_name": llm_cfg.get("llm_name", "unknown"),
            "pricing": pricing,
        },
        "overall": overall_stats,
        "per_instance": per_instance,
    }

    output_path = os.path.join(output_dir, "total_records.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(total_records, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[TotalRecords] Saved to {output_path}")
    logger.info(
        f"[TotalRecords] {overall_stats['resolved_count']}/{overall_stats['total_instances']} resolved "
        f"({overall_stats['resolved_rate']*100:.1f}%), "
        f"total_{cost_suffix}={overall_stats[f'total_{cost_suffix}']:.4f}, "
        f"avg_{cost_suffix}={overall_stats[f'avg_{cost_suffix}']:.6f}"
    )

    return total_records


def main():
    parser = argparse.ArgumentParser(
        description="并行运行SWE-bench评估 — CustomizedCodeAgent + PyCodeGraph"
    )
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="模型配置文件(yaml)")
    parser.add_argument("--max-steps", type=int, default=200, help="Agent最大步数")
    parser.add_argument("--agent-type", type=str, default="code", choices=["code", "meta"], help="Agent类型: code或meta")
    parser.add_argument("--max-steps-per-subagent", type=int, default=80, help="Meta agent每个sub-agent的最大步数")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default="http://sys-proxy-rd-relay.byted.org:8118", help="HTTP 代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1,bytedance.net,byted.org", help="No proxy")
    parser.add_argument("--skill-path", type=str, default=None, help="Skill目录路径，若指定则自动启用skill")
    parser.add_argument(
        "--initial-workspaces-dir",
        type=str,
        default="./_tmp/workspacces",
        help=(
            "保存每个case刚clone并checkout到base_commit后的初始workspace目录；"
            "默认保存到 ./_tmp/workspacces，设置为空字符串可关闭。"
        ),
    )
    parser.add_argument(
        "--use-codegraph",
        action="store_true",
        help=(
            "启用原始 CodeGraph（npm 版）。"
            "clone repo 后在容器内安装 @colbymchenry/codegraph 并构建 .codegraph/ 索引，"
            "同时为 agent 注册 codegraph_* 工具。"
        ),
    )
    parser.add_argument(
        "--use-new-code-graph",
        action="store_true",
        help=(
            "启用 CodeGraphPy（Python专用）；"
            "构建 .codegraph_py/ 索引，提供行/列精确位置信息。"
        ),
    )
    parser.add_argument(
        "--use-pycodegraph",
        action="store_true",
        help=(
            "启用 PyCodeGraph（本项目新集成的零依赖 Python code graph）；"
            "从 /home/fanmeihao/projects/PyCodeGraph 拷贝安装到容器，"
            "构建 .pycodegraph/graph.json 索引并为 agent 注册 pycodegraph_* 工具。"
            "与 --use-codegraph / --use-new-code-graph 互斥，本参数优先级最高。"
        ),
    )
    args = parser.parse_args()

    # 解析并校验 code graph 类型（三选一，若都设置则以 pycodegraph 优先）
    code_graph_kind = _resolve_code_graph_kind(args)
    if code_graph_kind == CODE_GRAPH_KIND_PYCODEGRAPH:
        enabled_msg = "PyCodeGraph"
    elif code_graph_kind == CODE_GRAPH_KIND_NEW_PY:
        enabled_msg = "CodeGraphPy (new)"
    elif code_graph_kind == CODE_GRAPH_KIND_OLD:
        enabled_msg = "CodeGraph (original)"
    else:
        enabled_msg = "None"

    args.exp_config = _resolve_path(args.exp_config)

    with open(args.exp_config, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/{args.agent_type}_agent_pycodegraph_limit{args.eval_limit}_{exp_name}_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.initial_workspaces_dir:
        args.initial_workspaces_dir = _resolve_path(args.initial_workspaces_dir)
        os.makedirs(args.initial_workspaces_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    if os.path.isfile(args.exp_config):
        shutil.copy2(args.exp_config, os.path.join(config_snapshot_dir, os.path.basename(args.exp_config)))

    run_config_record = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "max_steps": args.max_steps,
        "agent_type": args.agent_type,
        "max_steps_per_subagent": args.max_steps_per_subagent,
        "parallel": args.parallel,
        "output_dir": args.output_dir,
        "initial_workspaces_dir": args.initial_workspaces_dir,
        "code_graph_kind": code_graph_kind,
        "use_codegraph": args.use_codegraph,
        "use_new_code_graph": args.use_new_code_graph,
        "use_pycodegraph": args.use_pycodegraph,
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)
    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}, CodeGraph: {enabled_msg}")
    logger.info(f"Main log file: {main_log_path}")

    runner_config = {
        "exp_cfg": llm_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": "./src/prompts/query.j2",
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "max_steps": args.max_steps,
        "use_codegraph": (code_graph_kind == CODE_GRAPH_KIND_OLD),
        "use_new_code_graph": (code_graph_kind == CODE_GRAPH_KIND_NEW_PY),
        "use_pycodegraph": (code_graph_kind == CODE_GRAPH_KIND_PYCODEGRAPH),
    }

    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances,
    )
    logger.info(f"Total instances: {len(instances)}")

    tasks = [
        {
            "instance": inst,
            "llm_cfg": llm_cfg,
            "max_steps": args.max_steps,
            "tmp_root": args.output_dir,
            "runner_config": runner_config,
            "agent_type": args.agent_type,
            "max_steps_per_subagent": args.max_steps_per_subagent,
            "skill_path": args.skill_path,
            "code_graph_kind": code_graph_kind,
            "initial_workspaces_dir": args.initial_workspaces_dir,
        }
        for inst in instances
    ]

    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))

    if args.parallel <= 1:
        for task in tasks:
            result = run_single_instance(task)
            results.append(result)
            _log_eval_progress(result, results, len(instances))
    else:
        executor = ProcessPoolExecutor(max_workers=args.parallel)
        try:
            futures = {executor.submit(run_single_instance, task): task for task in tasks}
            pool_broken = False
            for future in as_completed(futures):
                task = futures[future]
                instance_id = task["instance"]["instance_id"]
                try:
                    result = future.result(timeout=worker_timeout)
                except TimeoutError:
                    logger.error(f"[Worker] Timeout for {instance_id} after {worker_timeout}s")
                    result = {"instance_id": instance_id, "error": "timeout", "resolved": False}
                except Exception as e:
                    logger.error(f"[Worker] Future failed for {instance_id}: {e}")
                    result = {"instance_id": instance_id, "error": str(e), "resolved": False}
                    err_msg = str(e).lower()
                    if "process pool" in err_msg and "terminated" in err_msg:
                        pool_broken = True
                results.append(result)
                _log_eval_progress(result, results, len(instances))
                if pool_broken:
                    logger.error(
                        "[Worker] Process pool broken — marking remaining futures as errored"
                    )
                    for f, t in futures.items():
                        if not f.done():
                            fid = t["instance"]["instance_id"]
                            results.append({
                                "instance_id": fid,
                                "error": "process pool terminated",
                                "resolved": False,
                            })
                            _log_eval_progress(results[-1], results, len(instances))
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    resolved = sum(1 for r in results if r.get("resolved", False))
    total = len(results)
    if total > 0:
        logger.info(f"Resolved: {resolved}/{total} ({resolved / total * 100:.1f}%)")

    logger.info(f"Results saved to {output_file}")

    generate_total_records(
        output_dir=args.output_dir,
        llm_cfg=llm_cfg,
        exp_config_name=os.path.basename(args.exp_config),
    )


if __name__ == "__main__":
    main()
