"""
OpenHands Baseline — 并行运行 SWE-bench 评估的脚本

支持三种模式：
  1. OpenHandsCodeAgent baseline（默认）
  2. OpenHandsCodeAgentPlanMode（--use-plan-mode）
  3. CodeAgentOpenHandsOptimized（--use-optimized-tools）

默认使用 doubao.yaml 配置运行。
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.funcs import render_j2
from src.agent.openhands_code_agent import OpenHandsCodeAgent
from src.agent.openhands_code_agent_plan_mode import OpenHandsCodeAgentPlanMode
from src.agent.code_agent_openhands_optimized import CodeAgentOpenHandsOptimized

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = str(REPO_ROOT / "_config" / "doubao.yaml")


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


class OpenHandsSweBenchRunner(SweBenchRunner):
    """SweBenchRunner 子类，使用 OpenHands baseline agents。"""

    def evaluate_instance(
        self,
        instance,
        workspace,
        run_id=None,
        swe_eval_timeout=1800,
    ):
        from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME
        from src.benchmarks.swe_bench_runner import (
            _format_command_error,
            _print_metrics_summary,
            _collect_extra_metrics,
            instance_log_context,
        )
        from openhands.workspace import DockerWorkspace

        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        instance_id = instance["instance_id"]
        base_commit = instance["base_commit"]
        repo_name = instance["repo"].split("/")[-1]
        repo_path = f"/workspace/{repo_name}"
        repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))

        log_dir = os.path.join(self.tmp_root, 'log', run_id)
        os.makedirs(log_dir, exist_ok=True)
        instance_log_path = os.path.join(log_dir, "instance_log.ansi")
        trajectory_path = os.path.join(log_dir, f"{instance_id}_trajectory.md")

        with instance_log_context(instance_log_path):
            logger.info(f"[{instance_id}] logging to {instance_log_path}")

            with open(trajectory_path, "w", encoding="utf-8") as f:
                f.write(f"# Trajectory: {instance_id}\n\n")

            def save_trajectory(record: dict):
                try:
                    with open(trajectory_path, "a", encoding="utf-8") as f:
                        role = record.get("role", "")
                        if role == "assistant":
                            f.write(f"## Assistant\n")
                            thinking = record.get("thinking", "")
                            if thinking:
                                f.write(f"**Thought**: {thinking}\n\n")
                        elif role == "tool":
                            f.write(f"## Tool\n")
                            f.write(f"**Tool**: {record.get('tool_name', '')}\n")
                            args = record.get("tool_args", {})
                            if args:
                                f.write(f"**Args**: ```json\n{json.dumps(args, indent=2)}\n```\n\n")
                            obs = record.get("observation", "")
                            if obs:
                                f.write(f"**Observation**: ```\n{obs[:2000]}\n```\n\n")
                        f.write("---\n\n")
                except Exception as ex:
                    logger.warning(f"Error saving trajectory: {ex}")

            repo_url = f"https://github.com/{instance['repo']}.git"
            logger.info(
                f"Preparing {repo_url} @ {base_commit} into {repo_path} "
                f"(timeout={repo_prepare_timeout}s)"
            )
            clone_result = workspace.execute_command(
                f"rm -rf {repo_path} && "
                f"git init {repo_path} && "
                f"cd {repo_path} && "
                f"git remote add origin {repo_url} && "
                f"git fetch --depth 1 origin {base_commit}",
                timeout=float(repo_prepare_timeout),
            )
            if clone_result.exit_code != 0:
                error_message = _format_command_error(clone_result)
                logger.error(f"git fetch failed for {instance_id}: {error_message}")
                raise RuntimeError(
                    f"Failed to fetch repository for {instance_id}: {error_message}"
                )

            checkout_result = workspace.execute_command(
                f"cd {repo_path} && git checkout --detach FETCH_HEAD",
                timeout=120.0,
            )
            if checkout_result.exit_code != 0:
                error_message = _format_command_error(checkout_result)
                logger.error(f"git checkout failed for {instance_id}: {error_message}")
                raise RuntimeError(
                    f"Failed to checkout repository for {instance_id}: {error_message}"
                )

            instance["repo_path"] = repo_path

            task_description = render_j2(
                template_name=os.path.basename(self.prompt_path),
                context={
                    "repo_path": repo_path,
                    "problem_statement": str(instance.get('problem_statement', '')).strip(),
                    "base_commit": base_commit
                }
            )

            if self.use_plan_mode:
                logger.info(f"Using OpenHandsCodeAgentPlanMode for {instance_id}")
                code_agent = OpenHandsCodeAgentPlanMode(
                    planner_cfg=self.cheap_exp_cfg,
                    executor_llm=None,
                )
                agent_result = code_agent.run(
                    instruction=task_description,
                    workspace=workspace,
                    callbacks=[save_trajectory],
                    repo_path=repo_path,
                    output_dir=log_dir,
                    planner_trajectory_path=os.path.join(
                        log_dir, f"{instance_id}_planner_trajectory.md"
                    ),
                    execution_trajectory_path=os.path.join(
                        log_dir, f"{instance_id}_execution_trajectory.md"
                    ),
                )
            elif self.use_optimized_tools:
                from openhands.sdk import LLM
                llm = LLM(
                    model=self.exp_cfg["llm_name"],
                    api_key=self.exp_cfg.get("key"),
                    base_url=self.exp_cfg.get("openai_base_url"),
                    api_version=self.exp_cfg.get("api_version"),
                )
                logger.info(f"Using CodeAgentOpenHandsOptimized for {instance_id}")
                code_agent = CodeAgentOpenHandsOptimized(
                    llm=llm,
                    repo_path=repo_path,
                )
                agent_result = code_agent.run(
                    instruction=task_description,
                    workspace=workspace,
                    callbacks=[save_trajectory],
                )
            else:
                logger.info(f"Using OpenHandsCodeAgent for {instance_id}")
                code_agent = OpenHandsCodeAgent(
                    llm_cfg=self.exp_cfg,
                    repo_path=repo_path,
                    base_commit=base_commit,
                )
                agent_result = code_agent.run(
                    instruction=task_description,
                    workspace=workspace,
                    callbacks=[save_trajectory],
                    output_dir=log_dir,
                )

            _print_metrics_summary(f"{instance_id}", agent_result.metrics)

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

            if not git_patch:
                logger.warning(f"No git patch generated for {instance_id}")

            logger.info(f"Running SWE-bench eval for {instance_id}...")

            from src.benchmarks.swebench.swe_eval import run_swebench_eval

            eval_result = run_swebench_eval(
                instance=instance,
                git_patch=git_patch,
                run_id=run_id,
                tmp_dir=self.tmp_root,
                timeout=swe_eval_timeout,
            )

            logger.info(
                f"[{instance_id}] resolved={eval_result['resolved']} "
                f"patch_applied={eval_result['patch_applied']}"
            )

            result = {
                "instance_id": instance_id,
                "git_patch": git_patch,
                "metrics": agent_result.metrics,
                **eval_result,
            }

            result.update(_collect_extra_metrics(agent_result.metrics))

            result_file = os.path.join(log_dir, f"{instance_id}_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {result_file}")

            return result


def run_single_instance(args_dict):
    instance = args_dict['instance']
    runner_config = args_dict['runner_config']

    instance_id = instance['instance_id']
    logger.info(f"[Worker] Starting {instance_id}")

    workspace = None
    try:
        runner = OpenHandsSweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)
        result = runner.evaluate_instance(instance, workspace)
        logger.info(f"[Worker] Completed {instance_id}: resolved={result.get('resolved', False)}")
        return result
    except Exception as e:
        logger.error(f"[Worker] Error in {instance_id}: {e}")
        return {
            "instance_id": instance_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "resolved": False
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Cleanup failed for {instance_id}: {cleanup_err}")


def main():
    parser = argparse.ArgumentParser(description="OpenHands Baseline — 并行运行SWE-bench评估")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, default=DEFAULT_CONFIG, help="主模型配置文件")
    parser.add_argument("--cheap-config", type=str, default=DEFAULT_CONFIG, help="便宜模型配置文件")
    parser.add_argument("--use-plan-mode", action="store_true", help="使用 OpenHandsCodeAgentPlanMode")
    parser.add_argument("--use-optimized-tools", action="store_true", help="使用 CodeAgentOpenHandsOptimized（优化工具集）")
    parser.add_argument("--prompt-path", type=str, default="./src/prompts/query.j2", help="Prompt模板")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default=None, help="HTTP代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1", help="No proxy")
    args = parser.parse_args()

    args.exp_config = _resolve_path(args.exp_config)
    args.cheap_config = _resolve_path(args.cheap_config)
    args.prompt_path = _resolve_path(args.prompt_path, fallback_base=REPO_ROOT / "src" / "prompts")

    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    with open(args.cheap_config, "r", encoding="utf-8") as f:
        cheap_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        if args.use_plan_mode:
            mode_tag = "openhands_plan_mode"
        elif args.use_optimized_tools:
            mode_tag = "openhands_optimized_tools"
        else:
            mode_tag = "openhands_code_agent"
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = (
            f"./_tmp/{mode_tag}_limit{args.eval_limit}"
            f"_{exp_name}"
            f"_{timestamp}"
        )
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    import shutil
    for cfg_path in [args.exp_config, args.cheap_config]:
        if cfg_path and os.path.isfile(cfg_path):
            dst = os.path.join(config_snapshot_dir, os.path.basename(cfg_path))
            shutil.copy2(cfg_path, dst)
    run_config_record = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "cheap_config": os.path.basename(args.cheap_config),
        "parallel": args.parallel,
        "use_plan_mode": args.use_plan_mode,
        "use_optimized_tools": args.use_optimized_tools,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "output_dir": args.output_dir,
        "agent_type": (
            "openhands_plan_mode" if args.use_plan_mode
            else "openhands_optimized_tools" if args.use_optimized_tools
            else "openhands_code_agent"
        ),
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)

    logger.info(
        f"Output: {args.output_dir}, Parallel: {args.parallel}, "
        f"PlanMode: {args.use_plan_mode}, OptimizedTools: {args.use_optimized_tools}, "
        f"Agent: {'OpenHandsCodeAgentPlanMode' if args.use_plan_mode else 'CodeAgentOpenHandsOptimized' if args.use_optimized_tools else 'OpenHandsCodeAgent'}"
    )
    logger.info(f"Main log file: {main_log_path}")

    runner_config = {
        "exp_cfg": exp_cfg,
        "cheap_exp_cfg": cheap_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": args.prompt_path,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "use_plan_mode": args.use_plan_mode,
        "use_optimized_tools": args.use_optimized_tools,
    }
    logger.info(f"Runner config: {runner_config}")

    runner = OpenHandsSweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances
    )
    logger.info(f"Total instances: {len(instances)}")
    logger.info(f"The instances are: {[inst['instance_id'] for inst in instances]}")

    tasks = [{"instance": inst, "runner_config": runner_config} for inst in instances]

    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_single_instance, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            instance_id = task["instance"]["instance_id"]
            try:
                result = future.result(timeout=worker_timeout)
            except TimeoutError:
                logger.error(f"[Worker] Timeout for {instance_id} after {worker_timeout}s")
                result = {
                    "instance_id": instance_id,
                    "error": f"Worker timeout after {worker_timeout}s",
                    "resolved": False,
                }
                future.cancel()
            except Exception as e:
                logger.error(f"[Worker] Future failed for {instance_id}: {e}")
                result = {
                    "instance_id": instance_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "resolved": False,
                }
            results.append(result)
            logger.info(f"Progress: {len(results)}/{len(instances)}")

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    resolved = sum(1 for r in results if r.get("resolved", False))
    if results:
        logger.info(f"Resolved: {resolved}/{len(results)} ({resolved/len(results)*100:.1f}%)")
    else:
        logger.info("Resolved: 0/0 (0.0%)")


if __name__ == "__main__":
    main()


"""
# ============================================================
# OpenHands Baseline CodeAgent — doubao
# ============================================================
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# OpenHands Baseline PlanMode — doubao
# ============================================================
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-plan-mode \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# OpenHands Optimized Tools — doubao
# ============================================================
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-optimized-tools \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# Quick test — single instance
# ============================================================
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 1 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
  

# OpenHands Baseline CodeAgent（默认 doubao.yaml）
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --parallel 16

# OpenHands Baseline PlanMode
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --parallel 16 \
  --use-plan-mode

# OpenHands Optimized Tools
python example/baseline/benchmark_openhands_code_agent.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --parallel 16 \
  --use-optimized-tools
"""
