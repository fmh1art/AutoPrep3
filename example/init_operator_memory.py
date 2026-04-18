"""
Operator Pipeline Memory Initialization Script.

整体流程:
  Step 0: 使用 CodeAgent 分别用 A/B/C 三个 LLM backbone 运行 SWE-bench 实例，收集 trajectory
  Step 1: 从 trajectory 中提取 subtask 信息，构建 subtask dataset
  Step 2: 使用 LLM 总结每个 subtask 的类型、步骤模式、token 消耗模式
  Step 3: 使用 LLM 总结每个 LLM backbone 的输出模式
  Step 4: 将总结结果写入对应 backbone 的 CEMemorizer

用法:
  # Step 0: 收集 trajectory (使用 A/B/C 三个模型分别运行)
  python -m example.init_operator_memory collect-trajectories \\
    --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \\
    --split test \\
    --eval-limit 20 \\
    --output-dir _tmp/operator_trajectories \\
    --http-proxy http://sys-proxy-rd-relay.byted.org:8118

  # Step 1: 从 trajectory 提取 subtask dataset
  python -m example.init_operator_memory extract-subtasks \\
    --trajectory-dir _tmp/operator_trajectories \\
    --output-dir _tmp/operator_subtask_datasets

  # Step 2+3+4: 总结并初始化 memory
  python -m example.init_operator_memory init-memory \\
    --subtask-dir _tmp/operator_subtask_datasets \\
    --llm-config _config/doubao.yaml \\
    --memory-root _tmp/memory_operator_ce \\
    --workers 8
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.agent.ce_memorizer import CEMemorizer, _normalize_fluctuation_ratio
from src.module import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string
from src.agent.operator import LLMBackbone

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Step 0: Collect Trajectories
# ============================================================================

def _run_single_instance_for_trajectory(args_dict: dict) -> dict:
    instance = args_dict["instance"]
    backbone_key = args_dict["backbone_key"]
    backbone_cfg = args_dict["backbone_cfg"]
    runner_config = args_dict["runner_config"]
    output_dir = args_dict["output_dir"]

    instance_id = instance["instance_id"]
    workspace = None

    try:
        from src.benchmarks.swe_bench_runner import SweBenchRunner
        from src.agent.code_agent import CodeAgent
        from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME
        from src.tools.funcs import render_j2 as _render_j2

        runner = SweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)

        repo_name = instance["repo"].split("/")[-1]
        repo_path = f"/workspace/{repo_name}"

        task_description = _render_j2(
            template_name="query.j2",
            context={
                "repo_path": repo_path,
                "problem_statement": str(instance.get("problem_statement", "")).strip(),
                "base_commit": instance["base_commit"],
            },
        )

        agent = CodeAgent(
            llm_cfg=backbone_cfg,
            max_steps=50,
            max_retries_per_call=3,
        )

        case_dir = os.path.join(output_dir, backbone_key, instance_id)
        os.makedirs(case_dir, exist_ok=True)

        result = agent.run(
            instruction=task_description,
            workspace=workspace,
            output_dir=case_dir,
        )

        with open(os.path.join(case_dir, "instance.json"), "w", encoding="utf-8") as f:
            json.dump(instance, f, ensure_ascii=False, indent=2)
        with open(os.path.join(case_dir, "backbone_info.json"), "w", encoding="utf-8") as f:
            json.dump({"backbone_key": backbone_key, "backbone_cfg_path": args_dict.get("backbone_cfg_path", "")}, f, ensure_ascii=False, indent=2)

        logger.info(f"[{backbone_key}] {instance_id}: completed, metrics={result.metrics}")

        return {
            "instance_id": instance_id,
            "backbone_key": backbone_key,
            "case_dir": case_dir,
            "status": "success",
            "total_steps": result.metrics.get("total_steps", 0),
        }

    except Exception as e:
        logger.error(f"[{backbone_key}] {instance_id}: ERROR - {e}")
        traceback.print_exc()
        return {
            "instance_id": instance_id,
            "backbone_key": backbone_key,
            "status": "error",
            "error": str(e),
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception:
                pass


def collect_trajectories(args):
    from src.benchmarks.swe_bench_runner import SweBenchRunner

    if args.http_proxy:
        os.environ["HTTP_PROXY"] = args.http_proxy
        os.environ["HTTPS_PROXY"] = args.http_proxy
        logger.info(f"Set HTTP_PROXY={args.http_proxy}")
    if args.no_proxy:
        os.environ["NO_PROXY"] = args.no_proxy

    with open(args.planner_config, "r") as f:
        planner_cfg = yaml.safe_load(f)

    backbone_configs = {}
    config_dir = os.path.dirname(args.planner_config)
    for backbone in [LLMBackbone.A, LLMBackbone.B, LLMBackbone.C]:
        cfg_path = os.path.join(config_dir, backbone.config_filename)
        if os.path.exists(cfg_path):
            backbone_configs[backbone.value] = yaml.safe_load(open(cfg_path))
        else:
            logger.warning(f"Config not found for backbone {backbone.value}: {cfg_path}")

    if not backbone_configs:
        logger.error("No backbone configs found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    runner_config = {
        "exp_cfg": planner_cfg,
        "cheap_exp_cfg": planner_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": "./src/prompts/query.j2",
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
    }

    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
    )

    logger.info(f"Total instances: {len(instances)}, Backbones: {list(backbone_configs.keys())}")

    tasks = []
    for instance in instances:
        for backbone_key, backbone_cfg in backbone_configs.items():
            tasks.append({
                "instance": instance,
                "backbone_key": backbone_key,
                "backbone_cfg": backbone_cfg,
                "backbone_cfg_path": os.path.join(config_dir, LLMBackbone(backbone_key).config_filename),
                "runner_config": runner_config,
                "output_dir": args.output_dir,
            })

    logger.info(f"Total tasks: {len(tasks)}")

    results = []
    if args.parallel <= 1:
        for task in tasks:
            result = _run_single_instance_for_trajectory(task)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(_run_single_instance_for_trajectory, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=7200)
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    logger.error(f"Task failed: {e}")
                    results.append({
                        "instance_id": task["instance"]["instance_id"],
                        "backbone_key": task["backbone_key"],
                        "status": "error",
                        "error": str(e),
                    })

    summary_path = os.path.join(args.output_dir, "collection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Collection complete: {success}/{len(results)} successful")


# ============================================================================
# Step 1: Extract Subtask Datasets from Trajectories (Multi-Granularity)
# ============================================================================

def _load_trajectory(case_dir: str) -> tuple[list[dict], str, str, str] | None:
    snapshot_path = os.path.join(case_dir, "trajectory_snapshot.json")
    result_path = os.path.join(case_dir, "agent_result.json")

    trajectory = []
    if os.path.exists(snapshot_path):
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trajectory = data.get("trajectory", [])
    elif os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trajectory = data.get("trajectory", [])

    if not trajectory:
        return None

    backbone_info_path = os.path.join(case_dir, "backbone_info.json")
    backbone_key = "B"
    if os.path.exists(backbone_info_path):
        with open(backbone_info_path, "r") as f:
            backbone_info = json.load(f)
            backbone_key = backbone_info.get("backbone_key", "B")

    instance_id = os.path.basename(case_dir)
    instruction = ""

    instance_path = os.path.join(case_dir, "instance.json")
    if os.path.exists(instance_path):
        with open(instance_path, "r") as f:
            instance = json.load(f)
            instance_id = instance.get("instance_id", instance_id)
            instruction = instance.get("problem_statement", "")

    return trajectory, backbone_key, instance_id, instruction


def _build_trajectory_summary(trajectory: list[dict]) -> list[dict]:
    summary = []
    for i, step in enumerate(trajectory):
        tool_name = step.get("tool_name", "")
        if tool_name == "finish":
            summary.append({
                "index": i,
                "tool_name": "finish",
                "detail": step.get("finish_message", "")[:100],
            })
            continue

        args = step.get("tool_args", {})
        if tool_name == "bash":
            cmd = args.get("command", "")
            detail = cmd[:80]
        elif tool_name == "file_editor":
            command = args.get("command", "")
            path = args.get("path", "")
            detail = f"{command} {path}"[:80]
        else:
            detail = str(args)[:80]

        obs_len = len(step.get("observation", "") or "")
        summary.append({"index": i, "tool_name": tool_name, "detail": detail, "obs_len": obs_len})

    return summary


def _segment_with_llm(
    trajectory: list[dict],
    instruction: str,
    backbone_key: str,
    cfg: dict,
) -> dict | None:
    llm = SimpleAPICaller(
        llm_name=cfg["llm_name"],
        api_key=cfg["key"],
        base_url=cfg.get("openai_base_url"),
        api_version=cfg.get("api_version"),
    )

    backbone = LLMBackbone(backbone_key)
    traj_summary = _build_trajectory_summary(trajectory)

    prompt = render_j2("operator_segment_trajectory.j2", context={
        "instruction": instruction[:1000],
        "backbone_display": backbone.display_name,
        "total_steps": len(trajectory),
        "trajectory_summary": traj_summary,
    })

    try:
        raw = llm.chat(prompt)
        parsed = parse_any_string(raw, code_type="json")
        result = json.loads(parsed)

        if not isinstance(result, dict):
            return None
        for granularity in ["fine", "medium", "coarse"]:
            if granularity not in result:
                return None
            if not isinstance(result[granularity], list):
                return None

        return result

    except Exception as e:
        logger.error(f"LLM segmentation failed: {e}")
        return None


def _segment_rule_based(trajectory: list[dict]) -> list[dict]:
    segments = []
    current_steps = []

    for i, step in enumerate(trajectory):
        tool_name = step.get("tool_name", "")
        current_steps.append(i)

        if tool_name == "finish":
            segments.append({
                "step_range": [current_steps[0], current_steps[-1]],
                "subtask_type": _classify_steps(trajectory, current_steps),
            })
            current_steps = []
        elif _is_rule_boundary(step, i, len(trajectory)):
            if len(current_steps) > 1:
                segments.append({
                    "step_range": [current_steps[0], current_steps[-2]],
                    "subtask_type": _classify_steps(trajectory, current_steps[:-1]),
                })
            current_steps = [i]
        else:
            pass

    if len(current_steps) > 0:
        segments.append({
            "step_range": [current_steps[0], current_steps[-1]],
            "subtask_type": _classify_steps(trajectory, current_steps),
        })

    return segments


def _is_rule_boundary(step: dict, index: int, total: int) -> bool:
    tool_name = step.get("tool_name", "")
    thinking = (step.get("thinking", "") or "").lower()

    if tool_name in ("bash", "file_editor"):
        observation = step.get("observation", "")
        if "test" in observation.lower() and ("pass" in observation.lower() or "fail" in observation.lower()):
            return True

    if thinking and any(kw in thinking for kw in ["now i will", "next, i", "let me now", "moving on", "now let"]):
        return True

    return False


def _classify_steps(trajectory: list[dict], step_indices: list[int]) -> str:
    has_edit = False
    has_test = False
    has_search = False

    for i in step_indices:
        if i >= len(trajectory):
            continue
        step = trajectory[i]
        tool_name = step.get("tool_name", "")
        args = step.get("tool_args", {})

        if tool_name == "file_editor" and args.get("command") in ("str_replace", "create", "insert"):
            has_edit = True
        if tool_name == "bash":
            cmd = args.get("command", "")
            if any(kw in cmd for kw in ["test", "pytest", "unittest"]):
                has_test = True
            if any(kw in cmd for kw in ["grep", "find", "search", "cat"]):
                has_search = True

    if has_edit and has_test:
        return "implement_and_verify"
    elif has_edit:
        return "implement"
    elif has_test:
        return "verify"
    else:
        return "explore"


def _build_subtask_record_from_range(
    trajectory: list[dict],
    step_range: list[int],
    subtask_type: str,
    instruction: str,
    backbone_key: str,
    instance_id: str,
    title: str = "",
    estimated_complexity: str = "medium",
    recommended_backbone: str = "B",
) -> dict:
    start, end = step_range
    steps = trajectory[start:end + 1]

    tool_calls = []
    output_tokens = []
    observation_tokens = []

    for step in steps:
        tool_name = step.get("tool_name", "")
        if tool_name == "finish":
            continue
        tool_calls.append(tool_name)
        usage = step.get("usage", {})
        output_tokens.append(usage.get("completion_tokens", 0))
        obs = step.get("observation", "")
        observation_tokens.append(len(obs) // 4 if obs else 0)

    thinking_parts = []
    for step in steps:
        t = step.get("thinking", "")
        if t:
            thinking_parts.append(t[:200])

    return {
        "instance_id": instance_id,
        "backbone_key": backbone_key,
        "title": title or subtask_type,
        "subtask_type": subtask_type,
        "step_range": step_range,
        "instruction": instruction[:500],
        "total_steps": len(steps),
        "tool_call_list": tool_calls,
        "output_token_list": output_tokens,
        "observation_token_list": observation_tokens,
        "trajectory_summary": "\n".join(thinking_parts)[:1000],
        "estimated_complexity": estimated_complexity,
        "recommended_backbone": recommended_backbone,
    }


def _extract_multigran_subtasks(
    case_dir: str,
    cfg: dict | None = None,
) -> dict[str, list[dict]]:
    loaded = _load_trajectory(case_dir)
    if loaded is None:
        return {"fine": [], "medium": [], "coarse": []}

    trajectory, backbone_key, instance_id, instruction = loaded
    result = {"fine": [], "medium": [], "coarse": []}

    llm_segmentation = None
    if cfg is not None:
        llm_segmentation = _segment_with_llm(trajectory, instruction, backbone_key, cfg)

    if llm_segmentation is not None:
        for granularity in ["fine", "medium", "coarse"]:
            for seg in llm_segmentation[granularity]:
                step_range = seg.get("step_range", [0, 0])
                subtask_type = seg.get("subtask_type", "explore")
                title = seg.get("title", subtask_type)
                complexity = seg.get("estimated_complexity", "medium")
                rec_backbone = seg.get("recommended_backbone", "B")

                record = _build_subtask_record_from_range(
                    trajectory, step_range, subtask_type, instruction,
                    backbone_key, instance_id, title, complexity, rec_backbone,
                )
                result[granularity].append(record)
    else:
        rule_segments = _segment_rule_based(trajectory)

        fine_segs = rule_segments
        medium_segs = _merge_segments(rule_segments, max_groups=4)
        coarse_segs = _merge_segments(rule_segments, max_groups=2)

        for seg in fine_segs:
            record = _build_subtask_record_from_range(
                trajectory, seg["step_range"], seg["subtask_type"],
                instruction, backbone_key, instance_id,
            )
            result["fine"].append(record)

        for seg in medium_segs:
            record = _build_subtask_record_from_range(
                trajectory, seg["step_range"], seg["subtask_type"],
                instruction, backbone_key, instance_id,
            )
            result["medium"].append(record)

        for seg in coarse_segs:
            record = _build_subtask_record_from_range(
                trajectory, seg["step_range"], seg["subtask_type"],
                instruction, backbone_key, instance_id,
            )
            result["coarse"].append(record)

    return result


def _merge_segments(segments: list[dict], max_groups: int = 3) -> list[dict]:
    if len(segments) <= max_groups:
        return segments

    n = len(segments)
    group_size = max(1, n // max_groups)
    merged = []

    for g in range(max_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, n)
        if g == max_groups - 1:
            end_idx = n

        group = segments[start_idx:end_idx]
        if not group:
            continue

        step_start = group[0]["step_range"][0]
        step_end = group[-1]["step_range"][1]

        types = [s["subtask_type"] for s in group]
        if "implement" in types or "implement_and_verify" in types:
            merged_type = "implement_and_verify" if "verify" in types else "implement"
        elif "verify" in types:
            merged_type = "verify"
        else:
            merged_type = "explore"

        merged.append({"step_range": [step_start, step_end], "subtask_type": merged_type})

    return merged


def extract_subtasks(args):
    trajectory_dir = args.trajectory_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = None
    if args.llm_config:
        cfg = yaml.safe_load(open(args.llm_config, "r"))

    all_subtasks: dict[str, dict[str, list[dict]]] = {}

    for backbone_key in ["A", "B", "C"]:
        backbone_dir = os.path.join(trajectory_dir, backbone_key)
        if not os.path.isdir(backbone_dir):
            logger.info(f"No trajectory directory for backbone {backbone_key}")
            continue

        case_dirs = sorted([
            os.path.join(backbone_dir, d)
            for d in os.listdir(backbone_dir)
            if os.path.isdir(os.path.join(backbone_dir, d))
        ])

        logger.info(f"Backbone {backbone_key}: found {len(case_dirs)} cases")

        backbone_subtasks = {"fine": [], "medium": [], "coarse": []}

        for case_dir in case_dirs:
            instance_id = os.path.basename(case_dir)
            logger.info(f"  Processing {instance_id}...")

            result = _extract_multigran_subtasks(case_dir, cfg=cfg)

            for granularity in ["fine", "medium", "coarse"]:
                backbone_subtasks[granularity].extend(result[granularity])

            fine_count = sum(len(result[g]) for g in ["fine", "medium", "coarse"])
            logger.info(f"    fine={len(result['fine'])}, medium={len(result['medium'])}, coarse={len(result['coarse'])}")

        all_subtasks[backbone_key] = backbone_subtasks

        for granularity in ["fine", "medium", "coarse"]:
            output_path = os.path.join(output_dir, f"subtask_dataset_{backbone_key}_{granularity}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(backbone_subtasks[granularity], f, ensure_ascii=False, indent=2)
            logger.info(f"  {granularity}: {len(backbone_subtasks[granularity])} subtasks -> {output_path}")

    total = sum(
        sum(len(v[g]) for g in ["fine", "medium", "coarse"])
        for v in all_subtasks.values()
    )
    logger.info(f"Total subtasks extracted: {total}")


# ============================================================================
# Step 2+3+4: Summarize and Initialize Memory
# ============================================================================

def _summarize_subtask_entry(subtask: dict, cfg: dict) -> dict | None:
    llm = SimpleAPICaller(
        llm_name=cfg["llm_name"],
        api_key=cfg["key"],
        base_url=cfg.get("openai_base_url"),
        api_version=cfg.get("api_version"),
    )

    try:
        subtask_title = subtask.get("title", subtask.get("subtask_type", "unknown"))
        plan_entry = [{"title": subtask_title, "step_ids": list(range(subtask["total_steps"]))}]

        output_token_list = subtask.get("output_token_list", [])
        observation_token_list = subtask.get("observation_token_list", [])

        if all(t == 0 for t in output_token_list) and subtask["total_steps"] > 0:
            output_token_list = [500] * subtask["total_steps"]

        if all(t == 0 for t in observation_token_list) and subtask["total_steps"] > 0:
            observation_token_list = [200] * subtask["total_steps"]

        prompt = render_j2("ce_memorize_subtask.j2", context={
            "instruction": subtask.get("instruction", ""),
            "plan": plan_entry,
            "subtask_index": 0,
            "total_steps": subtask["total_steps"],
            "trajectory": subtask.get("trajectory_summary", ""),
            "taken_tool_list": subtask.get("tool_call_list", []),
            "output_token_list": output_token_list,
            "observation_token_list": observation_token_list,
        })
        prompt = prompt.replace("[LAST_ERROR_PLACEHOLDER]\n\n", "")

        answer = llm.chat(prompt)
        parsed_answer = parse_any_string(answer, code_type="json")
        answer_dict = json.loads(parsed_answer)

        if not isinstance(answer_dict, dict):
            raise ValueError(f"Parsed answer is not a dictionary")
        if "subtask_type" not in answer_dict:
            raise ValueError(f"Missing 'subtask_type' key")
        if "knowledge" not in answer_dict:
            raise ValueError(f"Missing 'knowledge' key")
        if "indicators" not in answer_dict:
            raise ValueError(f"Missing 'indicators' key")

        indicators = answer_dict.get("indicators", {})
        if indicators.get("uncertainty_factor_exists") is True and "fluctuation_ratio" in indicators:
            _normalize_fluctuation_ratio(indicators)

        uncertainty_val = indicators.get("uncertainty")
        if uncertainty_val is not None:
            uncertainty_val = max(0.0, min(1.0, float(uncertainty_val)))
            indicators["uncertainty"] = round(uncertainty_val, 4)
        else:
            indicators["uncertainty"] = 0.5 if indicators.get("uncertainty_factor_exists") else 0.2

        parsed_answer = json.dumps(answer_dict, ensure_ascii=False)

        metric = llm.get_last_usage()
        token_metrics = {
            "input_token": metric.get("input_tokens", 0),
            "output_token": metric.get("output_tokens", 0),
            "cached_token": metric.get("cached_tokens", 0),
        }

        return {"summary_json": parsed_answer, "token_metrics": token_metrics}

    except Exception as e:
        logger.error(f"Failed to summarize subtask: {e}")
        return None


def _summarize_backbone_entry(backbone_key: str, subtasks: list[dict], cfg: dict) -> dict | None:
    llm = SimpleAPICaller(
        llm_name=cfg["llm_name"],
        api_key=cfg["key"],
        base_url=cfg.get("openai_base_url"),
        api_version=cfg.get("api_version"),
    )

    backbone = LLMBackbone(backbone_key)
    trajectory_lines = []
    for i, st in enumerate(subtasks[:20]):
        title = st.get("title", st.get("subtask_type", "unknown"))
        complexity = st.get("estimated_complexity", "medium")
        rec_bb = st.get("recommended_backbone", "B")
        trajectory_lines.append(
            f"Subtask {i}: title={title}, type={st.get('subtask_type', 'unknown')}, "
            f"complexity={complexity}, recommended_backbone={rec_bb}, "
            f"steps={st.get('total_steps', 0)}, tools={st.get('tool_call_list', [])}"
        )
    trajectory_str = "\n".join(trajectory_lines)

    instruction = subtasks[0].get("instruction", "") if subtasks else ""
    plan_str = f"Backbone {backbone_key} ({backbone.display_name}) execution pattern across {len(subtasks)} subtasks"

    try:
        prompt = render_j2("ce_memorize_backbone.j2", context={
            "instruction": instruction,
            "plan": plan_str,
            "llm_backbone_name": backbone.display_name,
            "trajectory": trajectory_str,
        })
        prompt = prompt.replace("[LAST_ERROR_PLACEHOLDER]\n\n", "")

        answer = llm.chat(prompt)
        parsed_answer = parse_any_string(answer, code_type="json")
        answer_dict = json.loads(parsed_answer)

        if not isinstance(answer_dict, dict):
            raise ValueError("Parsed answer is not a dictionary")
        if "summary" not in answer_dict:
            raise ValueError("Missing 'summary' key")

        parsed_answer = json.dumps(answer_dict, ensure_ascii=False)

        metric = llm.get_last_usage()
        token_metrics = {
            "input_token": metric.get("input_tokens", 0),
            "output_token": metric.get("output_tokens", 0),
            "cached_token": metric.get("cached_tokens", 0),
        }

        return {"summary_json": parsed_answer, "token_metrics": token_metrics, "backbone_key": backbone_key}

    except Exception as e:
        logger.error(f"Failed to summarize backbone {backbone_key}: {e}")
        return None


def _summarize_tool_entries(subtasks: list[dict], cfg: dict) -> list[dict]:
    llm = SimpleAPICaller(
        llm_name=cfg["llm_name"],
        api_key=cfg["key"],
        base_url=cfg.get("openai_base_url"),
        api_version=cfg.get("api_version"),
    )

    tool_samples: dict[str, list[dict]] = defaultdict(list)
    for st in subtasks:
        tool_call_list = st.get("tool_call_list", [])
        obs_token_list = st.get("observation_token_list", [])
        out_token_list = st.get("output_token_list", [])

        for i, tool_name in enumerate(tool_call_list):
            obs_tokens = obs_token_list[i] if i < len(obs_token_list) else 100
            out_tokens = out_token_list[i] if i < len(out_token_list) else 0

            if obs_tokens == 0:
                obs_tokens = 100

            if obs_tokens < 50:
                bucket = "small"
            elif obs_tokens < 500:
                bucket = "medium"
            elif obs_tokens < 2000:
                bucket = "large"
            else:
                bucket = "xlarge"

            key = f"{tool_name}_{bucket}"
            if len(tool_samples[key]) < 3:
                tool_samples[key].append({
                    "tool_name": tool_name,
                    "observation_token_count": obs_tokens,
                    "completion_tokens": out_tokens,
                    "bucket": bucket,
                })

    results = []
    for key, samples in tool_samples.items():
        for sample in samples:
            try:
                prompt = render_j2("ce_memorize_tools.j2", context={
                    "tool_name": sample["tool_name"],
                    "tool_input": f"(from {sample['bucket']} observation bucket)",
                    "tool_observation": f"(~{sample['observation_token_count']} tokens)",
                    "observation_token_count": sample["observation_token_count"],
                })
                prompt = prompt.replace("[LAST_ERROR_PLACEHOLDER]\n\n", "")

                answer = llm.chat(prompt)
                parsed_answer = parse_any_string(answer, code_type="json")
                answer_dict = json.loads(parsed_answer)

                if not isinstance(answer_dict, dict) or "summary" not in answer_dict:
                    continue

                parsed_answer = json.dumps(answer_dict, ensure_ascii=False)
                metric = llm.get_last_usage()
                token_metrics = {
                    "input_token": metric.get("input_tokens", 0),
                    "output_token": metric.get("output_tokens", 0),
                    "cached_token": metric.get("cached_tokens", 0),
                }

                results.append({
                    "tool_name": sample["tool_name"],
                    "summary_json": parsed_answer,
                    "token_metrics": token_metrics,
                })

            except Exception as e:
                logger.error(f"Failed to summarize tool {key}: {e}")

    return results


def init_memory(args):
    cfg = yaml.safe_load(open(args.llm_config, "r"))
    subtask_dir = args.subtask_dir
    memory_root = args.memory_root
    granularity = getattr(args, "granularity", "medium")

    total_token_metrics = {"input_token": 0, "output_token": 0, "cached_token": 0}

    for backbone_key in ["A", "B", "C"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Initializing memory for backbone {backbone_key}")
        logger.info(f"{'='*60}")

        backbone_mem_dir = os.path.join(memory_root, f"backbone_{backbone_key}")
        os.makedirs(backbone_mem_dir, exist_ok=True)

        memorizer = CEMemorizer(cfg=cfg, memory_root=backbone_mem_dir)

        dataset_path = os.path.join(subtask_dir, f"subtask_dataset_{backbone_key}_{granularity}.json")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(subtask_dir, f"subtask_dataset_{backbone_key}_medium.json")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(subtask_dir, f"subtask_dataset_{backbone_key}.json")
        if not os.path.exists(dataset_path):
            logger.warning(f"No subtask dataset for backbone {backbone_key}")
            continue

        with open(dataset_path, "r", encoding="utf-8") as f:
            subtasks = json.load(f)

        logger.info(f"Backbone {backbone_key}: {len(subtasks)} subtasks to process")

        # Phase 1: Summarize and add tool memories
        logger.info(f"\n--- Phase 1: Tool memory for backbone {backbone_key} ---")
        tool_results = _summarize_tool_entries(subtasks, cfg)
        for tr in tool_results:
            try:
                result = memorizer.add_memory("environment", tr["summary_json"], use_judge=True, tool_name=tr["tool_name"])
                if result.get("token_metrics"):
                    for k in total_token_metrics:
                        total_token_metrics[k] += result["token_metrics"].get(k, 0)
                logger.info(f"  [tool] {tr['tool_name']}: added={result['added']}, reason={result['reason']}")
            except Exception as e:
                logger.error(f"  [tool] {tr['tool_name']}: add ERROR - {e}")

        # Phase 2: Summarize and add subtask memories
        logger.info(f"\n--- Phase 2: Subtask memory for backbone {backbone_key} ---")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_summarize_subtask_entry, st, cfg): i
                for i, st in enumerate(subtasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result is None:
                        continue
                    add_result = memorizer.add_memory("task", result["summary_json"], use_judge=True)
                    if add_result.get("token_metrics"):
                        for k in total_token_metrics:
                            total_token_metrics[k] += add_result["token_metrics"].get(k, 0)
                    logger.info(f"  [subtask {idx}] added={add_result['added']}, reason={add_result['reason']}")
                except Exception as e:
                    logger.error(f"  [subtask {idx}] ERROR - {e}")

        # Phase 3: Summarize and add backbone memory
        logger.info(f"\n--- Phase 3: Backbone memory for backbone {backbone_key} ---")
        backbone_result = _summarize_backbone_entry(backbone_key, subtasks, cfg)
        if backbone_result:
            try:
                backbone = LLMBackbone(backbone_key)
                add_result = memorizer.add_memory(
                    "backbone", backbone_result["summary_json"],
                    use_judge=True, llm_backbone_name=backbone.display_name,
                )
                if add_result.get("token_metrics"):
                    for k in total_token_metrics:
                        total_token_metrics[k] += add_result["token_metrics"].get(k, 0)
                logger.info(f"  [backbone] added={add_result['added']}, reason={add_result['reason']}")
            except Exception as e:
                logger.error(f"  [backbone] add ERROR - {e}")

        memorizer.save_memory()
        logger.info(f"\nBackbone {backbone_key} memory saved to {backbone_mem_dir}")
        logger.info(f"  task_memory: {len(memorizer.task_memory)} entries")
        logger.info(f"  backbone_memory: {len(memorizer.backbone_memory)} entries")
        logger.info(f"  environment_memory: {len(memorizer.envirment_memory)} entries")

    logger.info(f"\n{'='*60}")
    logger.info(f"Memory initialization complete!")
    logger.info(f"Total LLM tokens used: {json.dumps(total_token_metrics)}")
    logger.info(f"Memory root: {memory_root}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Operator Pipeline Memory Initialization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Step 0: collect-trajectories
    p0 = subparsers.add_parser("collect-trajectories", help="Step 0: Collect trajectories using A/B/C LLMs")
    p0.add_argument("--dataset", type=str, required=True)
    p0.add_argument("--split", type=str, default="test")
    p0.add_argument("--eval-limit", type=int, default=20)
    p0.add_argument("--planner-config", type=str, required=True)
    p0.add_argument("--output-dir", type=str, default="_tmp/operator_trajectories")
    p0.add_argument("--parallel", type=int, default=1)
    p0.add_argument("--http-proxy", type=str, default=None)
    p0.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1")

    # Step 1: extract-subtasks
    p1 = subparsers.add_parser("extract-subtasks", help="Step 1: Extract multi-granularity subtask datasets from trajectories")
    p1.add_argument("--trajectory-dir", type=str, required=True)
    p1.add_argument("--output-dir", type=str, default="_tmp/operator_subtask_datasets")
    p1.add_argument("--llm-config", type=str, default=None, help="LLM config for segmentation (if not provided, use rule-based only)")

    # Step 2+3+4: init-memory
    p2 = subparsers.add_parser("init-memory", help="Step 2+3+4: Summarize and initialize memory")
    p2.add_argument("--subtask-dir", type=str, required=True)
    p2.add_argument("--llm-config", type=str, required=True)
    p2.add_argument("--memory-root", type=str, default="_tmp/memory_operator_ce")
    p2.add_argument("--granularity", type=str, default="medium", choices=["fine", "medium", "coarse"], help="Subtask granularity to use")
    p2.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.command == "collect-trajectories":
        collect_trajectories(args)
    elif args.command == "extract-subtasks":
        extract_subtasks(args)
    elif args.command == "init-memory":
        init_memory(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
