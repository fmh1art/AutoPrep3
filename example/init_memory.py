"""
Memory initialization script.

Reads execution logs from completed cases and uses CEMemorizer to build
initial task / backbone / environment memories.

Usage:
    python -m example.init_memory [--log_dir LOG_DIR] [--llm_config LLM_CONFIG] [--memory_root MEMORY_ROOT]
"""

import argparse
import json
import os
import yaml
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent.ce_memorizer import CEMemorizer, _normalize_fluctuation_ratio
from src.module import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------

def build_subtask_trajectory(execution_steps: dict, subtask: dict) -> list:
    """Build trajectory for a subtask: initial instruction + each step's I/O."""
    steps = execution_steps.get("steps", [])
    initial_prompt = execution_steps.get("initial_prompt", "")
    step_ids = subtask.get("step_ids", [])

    trajectory = []
    trajectory.append({
        "role": "initial_task",
        "content": initial_prompt,
    })

    for sid in step_ids:
        if sid >= len(steps):
            continue
        step = steps[sid]
        entry = {"step_index": sid, "event_type": step.get("event_type", "")}

        if step.get("event_type") == "ActionEvent":
            action = step.get("action", {})
            entry["input"] = {
                "tool": action.get("tool", ""),
                "args": action.get("args", {}),
                "thought": step.get("thought", ""),
            }
        elif step.get("event_type") == "ObservationEvent":
            entry["output"] = step.get("observation", "")

        trajectory.append(entry)

    return trajectory


def build_backbone_trajectory(execution_steps: dict, token_usage: dict) -> list:
    """Build trajectory for LLM backbone: output content + token length per step."""
    steps = execution_steps.get("steps", [])
    per_step_tokens = token_usage.get("execution_per_step", [])

    # Build a map from response_id to token info
    resp_id_to_tokens = {}
    for t in per_step_tokens:
        resp_id_to_tokens[t["response_id"]] = t

    trajectory = []
    for step in steps:
        if step.get("event_type") != "ActionEvent":
            continue
        resp_id = step.get("llm_response_id", "")
        action = step.get("action", {})
        thought = step.get("thought", "")

        # Reconstruct output content from action
        output_content = ""
        if thought:
            output_content += thought + "\n"
        if action.get("tool"):
            output_content += f"[Tool Call: {action['tool']}]"
            if action.get("summary"):
                output_content += f" {action['summary']}"

        token_info = resp_id_to_tokens.get(resp_id, {})
        trajectory.append({
            "step_index": step.get("index"),
            "output_content": output_content,
            "completion_tokens": token_info.get("completion_tokens", 0),
            "reasoning_tokens": token_info.get("reasoning_tokens", 0),
            "answer_tokens": token_info.get("answer_tokens", 0),
        })

    return trajectory


def build_tool_trajectory(execution_steps: dict, token_usage: dict, max_obs_chars: int = 100) -> list:
    """Build trajectory for tool memory: input, output (truncated), observation token count."""
    steps = execution_steps.get("steps", [])
    per_step_tokens = token_usage.get("execution_per_step", [])

    # Map step_index to next observation
    obs_map = {}
    for i, step in enumerate(steps):
        if step.get("event_type") == "ObservationEvent":
            # Find the preceding ActionEvent
            for j in range(i - 1, -1, -1):
                if steps[j].get("event_type") == "ActionEvent":
                    obs_map[steps[j]["index"]] = step.get("observation", "")
                    break

    # Map response_id to per-step token info
    resp_id_to_tokens = {}
    for t in per_step_tokens:
        if t.get("step_index") is not None:
            resp_id_to_tokens[t["step_index"]] = t

    trajectory = []
    for step in steps:
        if step.get("event_type") != "ActionEvent":
            continue
        action = step.get("action", {})
        tool_name = action.get("tool", "")
        if not tool_name:
            continue

        step_idx = step.get("index")
        observation = obs_map.get(step_idx, "")
        obs_truncated = observation[:max_obs_chars] + "..." if len(observation) > max_obs_chars else observation

        # Estimate observation token count (rough: chars / 4)
        obs_token_count = len(observation) // 4

        # Get completion tokens for this step
        token_info = resp_id_to_tokens.get(step_idx, {})

        trajectory.append({
            "tool_name": tool_name,
            "tool_input": json.dumps(action.get("args", {}), ensure_ascii=False)[:500],
            "tool_observation": obs_truncated,
            "observation_token_count": obs_token_count,
            "completion_tokens": token_info.get("completion_tokens", 0),
        })

    return trajectory


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_case_data(case_dir: str):
    """Load all log files for a single case."""
    log_dir = os.path.join(case_dir, "log")
    data = {}

    for fname in ["execution_steps.json", "subtasks.json", "token_usage.json", "PLAN.md"]:
        fpath = os.path.join(log_dir, fname)
        if not os.path.exists(fpath):
            continue
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                data[fname.replace(".json", "")] = json.load(f)
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                data[fname.replace(".md", "")] = f.read()

    return data


def _summarize_case(case_dir: str, cfg: dict):
    """Summarize a single case using a thread-local SimpleAPICaller.

    Returns a dict containing all summarize results for the case,
    or None if the case should be skipped.
    """
    case_id = os.path.basename(case_dir)
    data = load_case_data(case_dir)

    if "execution_steps" not in data or "token_usage" not in data:
        print(f"  [{case_id}] Skipping: missing execution_steps or token_usage")
        return None

    llm = SimpleAPICaller(
        llm_name=cfg['llm_name'],
        api_key=cfg['key'],
        base_url=cfg['openai_base_url'],
        api_version=cfg.get('api_version', None),
    )

    exec_steps = data["execution_steps"]
    token_usage = data["token_usage"]
    subtasks_data = data.get("subtasks", {})
    plan_text = data.get("PLAN", "")
    instruction = exec_steps.get("initial_prompt", "")

    subtasks = subtasks_data.get("subtasks", [])
    llm_backbone_name = "doubao"
    if token_usage.get("total", {}).get("token_usages"):
        llm_backbone_name = token_usage["total"]["token_usages"][0].get("model", "doubao")

    print(f"  [{case_id}] Processing {len(subtasks)} subtasks ...")

    case_token_metrics = {"input_token": 0, "output_token": 0, "cached_token": 0}

    task_results = []
    for si, subtask in enumerate(subtasks):
        step_ids = subtask.get("step_ids", [])
        traj = build_subtask_trajectory(exec_steps, subtask)

        taken_tools, output_tokens, obs_tokens = [], [], []
        steps = exec_steps.get("steps", [])
        for sid in step_ids:
            if sid >= len(steps):
                continue
            s = steps[sid]
            if s.get("event_type") == "ActionEvent":
                tool = s.get("action", {}).get("tool", "")
                if tool:
                    taken_tools.append(tool)
                ct = 0
                resp_id = s.get("llm_response_id", "")
                for t in token_usage.get("execution_per_step", []):
                    if t.get("response_id") == resp_id:
                        ct = t.get("completion_tokens", 0)
                        break
                output_tokens.append(ct)
            elif s.get("event_type") == "ObservationEvent":
                obs_text = s.get("observation", "")
                obs_tokens.append(len(obs_text) // 4)

        last_error = None
        for sid in reversed(step_ids):
            if sid < len(steps) and steps[sid].get("event_type") == "AgentErrorEvent":
                last_error = "AgentErrorEvent encountered"
                break

        try:
            prompt = render_j2('ce_memorize_subtask.j2', context={
                "instruction": instruction,
                "plan": subtasks,
                "subtask_index": si,
                "total_steps": len(step_ids),
                "trajectory": traj,
                "taken_tool_list": taken_tools,
                "output_token_list": output_tokens,
                "observation_token_list": obs_tokens,
            })
            if last_error:
                prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]', f"The last error encountered is: {last_error}")
            else:
                prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

            answer = llm.chat(prompt)
            parsed_answer = parse_any_string(answer, code_type='json')
            answer_dict = json.loads(parsed_answer)

            if not isinstance(answer_dict, dict):
                raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
            if 'knowledge' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'knowledge' key: {answer_dict}")
            if 'indicators' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'indicators' key: {answer_dict}")
            if answer_dict['indicators'].get('uncertainty_factor_exists') is True and 'fluctuation_ratio' not in answer_dict['indicators']:
                raise ValueError(f"Parsed answer's indicators indicate uncertainty factor exists but does not contain 'fluctuation_ratio' key: {answer_dict}")
            if answer_dict['indicators'].get('uncertainty_factor_exists') is True:
                _normalize_fluctuation_ratio(answer_dict['indicators'])
                parsed_answer = json.dumps(answer_dict, ensure_ascii=False)

            metric = llm.get_last_usage()
            tm = {
                'input_token': metric.get('input_tokens', 0),
                'output_token': metric.get('output_tokens', 0),
                'cached_token': metric.get('cached_tokens', 0),
            }
            _accumulate_tokens(case_token_metrics, tm)
            task_results.append({"subtask_index": si, "summary_json": parsed_answer, "token_metrics": tm})
            print(f"    [{case_id}] subtask {si}: summarized")
        except Exception as e:
            print(f"    [{case_id}] subtask {si}: ERROR - {e}")

    backbone_result = None
    backbone_traj = build_backbone_trajectory(exec_steps, token_usage)
    try:
        prompt = render_j2('ce_memorize_backbone.j2', context={
            "instruction": instruction,
            "plan": plan_text,
            "llm_backbone_name": llm_backbone_name,
            "trajectory": backbone_traj,
        })
        prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

        answer = llm.chat(prompt)
        parsed_answer = parse_any_string(answer, code_type='json')
        answer_dict = json.loads(parsed_answer)

        if not isinstance(answer_dict, dict):
            raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
        if 'summary' not in answer_dict:
            raise ValueError(f"Parsed answer does not contain 'summary' key: {answer_dict}")

        metric = llm.get_last_usage()
        tm = {
            'input_token': metric.get('input_tokens', 0),
            'output_token': metric.get('output_tokens', 0),
            'cached_token': metric.get('cached_tokens', 0),
        }
        _accumulate_tokens(case_token_metrics, tm)
        backbone_result = {"summary_json": parsed_answer, "llm_backbone_name": llm_backbone_name, "token_metrics": tm}
        print(f"    [{case_id}] backbone: summarized")
    except Exception as e:
        print(f"    [{case_id}] backbone: ERROR - {e}")

    tool_results = []
    tool_traj = build_tool_trajectory(exec_steps, token_usage)
    seen_tools = set()
    for item in tool_traj:
        tool_name = item["tool_name"]
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        try:
            prompt = render_j2('ce_memorize_tools.j2', context={
                "tool_name": tool_name,
                "tool_input": item["tool_input"],
                "tool_observation": item["tool_observation"],
                "observation_token_count": item["observation_token_count"],
            })
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

            answer = llm.chat(prompt)
            parsed_answer = parse_any_string(answer, code_type='json')
            answer_dict = json.loads(parsed_answer)

            if not isinstance(answer_dict, dict):
                raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
            if 'summary' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'summary' key: {answer_dict}")

            metric = llm.get_last_usage()
            tm = {
                'input_token': metric.get('input_tokens', 0),
                'output_token': metric.get('output_tokens', 0),
                'cached_token': metric.get('cached_tokens', 0),
            }
            _accumulate_tokens(case_token_metrics, tm)
            tool_results.append({"tool_name": tool_name, "summary_json": parsed_answer, "token_metrics": tm})
            print(f"    [{case_id}] tool [{tool_name}]: summarized")
        except Exception as e:
            print(f"    [{case_id}] tool [{tool_name}]: ERROR - {e}")

    return {
        "case_id": case_id,
        "task_results": task_results,
        "backbone_result": backbone_result,
        "tool_results": tool_results,
        "token_metrics": case_token_metrics,
    }


def init_memory(log_dir: str, llm_config_path: str, memory_root: str, max_workers: int = 4):
    cfg = yaml.safe_load(open(llm_config_path, "r"))
    memorizer = CEMemorizer(cfg=cfg, memory_root=memory_root)

    case_dirs = sorted([
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ])

    print(f"Found {len(case_dirs)} cases in {log_dir}")
    print(f"Using {max_workers} parallel workers for summarization")

    total_token_metrics = {"input_token": 0, "output_token": 0, "cached_token": 0}

    case_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(_summarize_case, case_dir, cfg): case_dir
            for case_dir in case_dirs
        }
        for future in as_completed(future_to_case):
            case_dir = future_to_case[future]
            case_id = os.path.basename(case_dir)
            try:
                result = future.result()
                if result is not None:
                    case_summaries.append(result)
            except Exception as e:
                print(f"  [{case_id}] Unexpected ERROR during summarization: {e}")

    case_summaries.sort(key=lambda x: x["case_id"])

    print(f"\nSummarization complete. Adding memories sequentially ...")

    for cs in case_summaries:
        case_id = cs["case_id"]
        _accumulate_tokens(total_token_metrics, cs["token_metrics"])

        for tr in cs["task_results"]:
            try:
                result = memorizer.add_memory("task", tr["summary_json"], use_judge=True)
                if result.get("token_metrics"):
                    _accumulate_tokens(total_token_metrics, result["token_metrics"])
                print(f"    [{case_id}] subtask {tr['subtask_index']}: added={result['added']}, reason={result['reason']}")
            except Exception as e:
                print(f"    [{case_id}] subtask {tr['subtask_index']}: add ERROR - {e}")

        if cs["backbone_result"]:
            br = cs["backbone_result"]
            try:
                result = memorizer.add_memory("backbone", br["summary_json"], use_judge=True, llm_backbone_name=br["llm_backbone_name"])
                if result.get("token_metrics"):
                    _accumulate_tokens(total_token_metrics, result["token_metrics"])
                print(f"    [{case_id}] backbone: added={result['added']}, reason={result['reason']}")
            except Exception as e:
                print(f"    [{case_id}] backbone: add ERROR - {e}")

        for tr in cs["tool_results"]:
            try:
                result = memorizer.add_memory("environment", tr["summary_json"], use_judge=True, tool_name=tr["tool_name"])
                if result.get("token_metrics"):
                    _accumulate_tokens(total_token_metrics, result["token_metrics"])
                print(f"    [{case_id}] tool [{tr['tool_name']}]: added={result['added']}, reason={result['reason']}")
            except Exception as e:
                print(f"    [{case_id}] tool [{tr['tool_name']}]: add ERROR - {e}")

    memorizer.save_memory()
    print(f"\nMemory saved to {memory_root}")
    print(f"Total LLM tokens used for memorization: {json.dumps(total_token_metrics)}")
    print(f"  task_memory: {len(memorizer.task_memory)} entries")
    print(f"  backbone_memory: {len(memorizer.backbone_memory)} entries")
    print(f"  environment_memory: {len(memorizer.envirment_memory)} entries")


def _accumulate_tokens(total: dict, metrics: dict):
    if not metrics:
        return
    for k in total:
        v = metrics.get(k, 0)
        if v > 0:
            total[k] += v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize CE memory from execution logs")
    parser.add_argument("--log_dir", default="_tmp/parallel_2026-04-08_19-07-48/log",
                        help="Root log directory containing case subdirectories")
    parser.add_argument("--llm_config", default="_config/doubao.yaml",
                        help="Path to LLM config YAML")
    parser.add_argument("--memory_root", default="_tmp/memory_ce",
                        help="Directory to store memory JSONL files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for summarization (default: 4)")
    args = parser.parse_args()

    init_memory(args.log_dir, args.llm_config, args.memory_root, max_workers=args.workers)

# python -m example.init_memory --log_dir _tmp/parallel_2026-04-08_19-07-48/log --llm_config _config/doubao.yaml --memory_root _tmp/memory_ce --workers 32