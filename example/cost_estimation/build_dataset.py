from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List


# 复用同目录下的 cal_cost 工具函数
try:
    from example.cost_estimation import cal_cost as cc
except Exception as e:  # pragma: no cover
    print(f"[ERROR] 无法导入 cal_cost: {e}", file=sys.stderr)
    raise


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    import re as _re
    return len(_re.findall(r"\w+|[^\w\s]", text, flags=_re.UNICODE))


def _parse_initial_prompt_meta(initial_prompt: str | None) -> dict:
    """从 initial_prompt 中解析 SweBenchRunner 构造的元数据。

    期望格式（见 src/benchmarks/swe_bench_runner.py:531 处构造的 task_description）：
      Instance: <id>
      Repo: <owner/repo>
      BaseCommit: <sha>
      RepoPath: <path>
      ProblemStatement:
      <multi-line ...>
    """
    meta = {
        "instance_id": None,
        "repo": None,
        "base_commit": None,
        "repo_path": None,
        "problem_statement": None,
    }
    if not initial_prompt:
        return meta

    lines = initial_prompt.splitlines()
    ps_idx = None
    ps_lines: List[str] = []
    for i, ln in enumerate(lines):
        if ln.startswith("Instance:"):
            meta["instance_id"] = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("Repo:"):
            meta["repo"] = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("BaseCommit:"):
            meta["base_commit"] = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("RepoPath:"):
            meta["repo_path"] = ln.split(":", 1)[1].strip() or None
        elif ln.strip() == "ProblemStatement:":
            ps_idx = i
            break

    if ps_idx is not None:
        ps_lines = lines[ps_idx + 1 :]
        meta["problem_statement"] = "\n".join(ps_lines).strip() or None

    return meta


def _pair_turns_in_subtask(
    step_ids: List[int],
    idx_to_step: Dict[int, dict],
    idx_to_completion: Dict[int, int],
    case: dict,
) -> List[dict]:
    """将一个 subtask 内的 indices 配对为若干回合：Action(输出) → Observation/AgentError。

    增强：对于错误事件（AgentError/EnvironmentError/ToolError），当缺失可用 observation 文本时，
    使用“下一次 ActionEvent 的 prompt_tokens 与当前 ActionEvent 的 prompt_tokens 差值”来估算该次
    错误观测带来的 prompt 开销，并记入 observation_tokens（不小于 0）。
    """
    turns: List[dict] = []
    ordered = sorted({int(i) for i in step_ids})
    # 所有 step 的全局有序索引（用于跨 subtask 搜索下一次 ActionEvent）
    all_indices = sorted(int(k) for k in idx_to_step.keys())
    for i, idx in enumerate(ordered):
        s = idx_to_step.get(idx)
        if not s or s.get("event_type") != "ActionEvent":
            continue

        # 输出内容优先构造成“工具名 + 完整参数”的规范描述，避免仅保留工具名
        action = s.get("action") or {}
        tool_name = action.get("tool")
        aargs = action.get("args")
        acontent = action.get("content")
        resp_id = s.get("llm_response_id") or action.get("response_id")
        output_text = ""
        try:
            if tool_name or aargs is not None or acontent:
                parts: list[str] = []
                if tool_name:
                    parts.append(f"tool={tool_name}")
                if aargs is not None:
                    if isinstance(aargs, dict):
                        parts.append("args=" + json.dumps(aargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    else:
                        parts.append("args=" + str(aargs))
                if acontent not in (None, ""):
                    parts.append("content=" + (acontent if isinstance(acontent, str) else str(acontent)))
                if resp_id:
                    parts.append(f"response_id={resp_id}")
                output_text = " | ".join(parts)
        except Exception:
            # 构造失败则回退
            output_text = ""

        # 若仍为空，再用其他信号兜底
        if not output_text:
            output_text = (
                s.get("llm_response_text")
                or action.get("summary")
                or action.get("content")
                or ""
            )
        if not output_text:
            # 最终兜底：终端/编辑器等工具的参数或完整 action JSON
            aargs2 = action.get("args") or {}
            if isinstance(aargs2, dict):
                cmd = aargs2.get("cmd") or aargs2.get("command") or aargs2.get("commands")
                if cmd:
                    output_text = str(cmd)
            if not output_text:
                output_text = json.dumps(action, ensure_ascii=False) if action else ""

        # 输出 token 口径：output_tokens = answer_tokens + reasoning_tokens
        # 先取并存字段；若缺失则回退到 completion 作为 answer，reasoning=0
        ans_t, rea_t = (0, 0)
        try:
            ans_t, rea_t = _get_answer_reasoning_tokens_for_step(case, idx)
        except Exception:
            ans_t, rea_t = (0, 0)
        if ans_t == 0 and rea_t == 0:
            ans_t = int(idx_to_completion.get(idx, 0))
            rea_t = 0
        output_tokens = int(ans_t) + int(rea_t)

        # 查找该 Action 之后的第一个观测/错误事件（限制在本 subtask 的 step_ids 范围内）
        obs_index = None
        observation_text = ""
        observation_tokens = 0
        error_flag = False

        for j in ordered:
            if j <= idx:
                continue
            sj = idx_to_step.get(j) or {}
            et = sj.get("event_type")
            if et in ("ObservationEvent", "AgentErrorEvent", "EnvironmentErrorEvent", "ToolErrorEvent"):
                obs_index = j
                if et == "ObservationEvent":
                    observation_text = sj.get("observation") or ""
                    observation_tokens = _approx_token_count(observation_text)
                else:
                    # 错误事件：尽量取可读信息；若仍为空，使用固定占位文本
                    error_flag = True
                    observation_text = sj.get("message") or sj.get("error") or ""
                    if not observation_text:
                        observation_text = "Error Action!"
                    # 1) 以文本近似作为下限（保留原有估算）
                    txt_tokens = _approx_token_count(observation_text) if observation_text else 0
                    # 2) 基于 prompt_tokens 差值估算错误观测带来的实际 prompt 成本
                    try:
                        curr_pt = _get_prompt_tokens_for_step(case, idx)
                    except Exception:
                        curr_pt = 0
                    next_action_idx = None
                    for k in all_indices:
                        if k > idx and (idx_to_step.get(k) or {}).get("event_type") == "ActionEvent":
                            next_action_idx = k
                            break
                    if next_action_idx is not None:
                        try:
                            next_pt = _get_prompt_tokens_for_step(case, next_action_idx)
                        except Exception:
                            next_pt = 0
                        delta = next_pt - curr_pt
                        if delta < 0:
                            delta = 0
                        # 使用差值覆盖为主要估算；若差值为 0，则保留文本估算
                        observation_tokens = delta if delta > 0 else txt_tokens
                    else:
                        observation_tokens = txt_tokens
                break

        turns.append(
            {
                "action_index": idx,
                "output": output_text,
                "output_tokens": output_tokens,
                "answer_tokens": int(ans_t),
                "reasoning_tokens": int(rea_t),
                "observation_index": obs_index,
                "observation": observation_text if observation_text is not None else "",
                "observation_tokens": observation_tokens,
                "observation_error": error_flag,
            }
        )
    return turns


def _get_prompt_tokens_for_step(case: dict, step_index: int) -> int:
    """获取指定 Action step 的 prompt token 数，优先 execution_per_step；否则用 response_id 回退。"""
    tu = (case or {}).get("token_usage") or {}
    per_step = tu.get("execution_per_step")
    if isinstance(per_step, list):
        for item in per_step:
            try:
                idx = int(item.get("step_index")) if item.get("step_index") is not None else None
            except Exception:
                idx = None
            if idx == step_index:
                return int(item.get("prompt_tokens", 0) or 0)

    # 回退：通过 response_id 匹配
    es = (case or {}).get("execution_steps") or {}
    steps = es.get("steps") or []
    rid = None
    for s in steps:
        try:
            if int(s.get("index")) == int(step_index) and s.get("event_type") == "ActionEvent":
                rid = s.get("llm_response_id") or (s.get("action") or {}).get("response_id")
                break
        except Exception:
            continue
    if rid is not None:
        exec_block = (tu.get("execution") or {})
        for u in (exec_block.get("token_usages") or []):
            if str(u.get("response_id", "")) == str(rid):
                return int(u.get("prompt_tokens", 0) or 0)
    return 0


def _get_answer_reasoning_tokens_for_step(case: dict, step_index: int) -> tuple[int, int]:
    """返回 (answer_tokens, reasoning_tokens)。

    优先读取 token_usage.execution_per_step[*]；若不存在，回退到
    token_usage.execution.token_usages 通过 response_id 匹配；再不行则 (0,0)。
    """
    tu = (case or {}).get("token_usage") or {}
    per_step = tu.get("execution_per_step")
    if isinstance(per_step, list):
        for item in per_step:
            try:
                idx = int(item.get("step_index")) if item.get("step_index") is not None else None
            except Exception:
                idx = None
            if idx == step_index:
                ans = int(item.get("answer_tokens", item.get("completion_tokens", 0)) or 0)
                rea = int(item.get("reasoning_tokens", 0) or 0)
                return ans, rea

    es = (case or {}).get("execution_steps") or {}
    steps = es.get("steps") or []
    rid = None
    for s in steps:
        try:
            if int(s.get("index")) == int(step_index) and s.get("event_type") == "ActionEvent":
                rid = s.get("llm_response_id") or (s.get("action") or {}).get("response_id")
                break
        except Exception:
            continue
    exec_block = (tu.get("execution") or {})
    if rid is not None:
        for u in (exec_block.get("token_usages") or []):
            if str(u.get("response_id", "")) == str(rid):
                ans = int(u.get("completion_tokens", 0) or 0)
                rea = int(u.get("reasoning_tokens", 0) or 0)
                return ans, rea
    return 0, 0


def _compute_prefix_tokens_first_action(case: dict, subtasks: list[dict], idx_to_step: Dict[int, dict]) -> dict:
    """计算第一个 subtask 的第一个 ActionEvent 之前的 prompt token 数（即该 Action 的 input prompt 长度）。"""
    if not subtasks:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    ids_raw = subtasks[0].get("step_ids") or []
    step_ids: List[int] = []
    for x in ids_raw:
        try:
            step_ids.append(int(x))
        except Exception:
            continue
    for i in sorted(set(step_ids)):
        s = idx_to_step.get(i)
        if s and s.get("event_type") == "ActionEvent":
            pt = _get_prompt_tokens_for_step(case, i)
            return {"prompt_tokens": pt, "completion_tokens": 0, "total_tokens": pt}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def build_dataset(tmp_root: str, subdir: str, out_dir: str | None = None) -> str:
    """构造数据集并写入 JSONL，返回输出文件路径。"""
    cases = cc.load_parallel_cases(tmp_root, subdir)
    if out_dir is None:
        out_dir = os.path.join(tmp_root, subdir, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "subtasks_dataset.jsonl")

    with open(out_path, "w", encoding="utf-8") as outf:
        for case in cases:
            es = (case or {}).get("execution_steps") or {}
            if not es:
                continue
            initial_prompt = es.get("initial_prompt") or None
            meta = _parse_initial_prompt_meta(initial_prompt)

            # 优先从 result.json 获取 instance_id 等信息
            result = case.get("result") or {}
            if result.get("instance_id"):
                meta["instance_id"] = result.get("instance_id")

            # 基础映射
            idx_to_step, idx_to_completion, idx_to_obs = cc._collect_step_mappings(case)

            # 加载 subtasks（优先分类后的；否则原始）
            paths = (case or {}).get("_paths") or {}
            es_path = paths.get("execution_steps")
            case_root = os.path.dirname(es_path) if es_path else (paths.get("case_root") or "")
            # 在元数据中保留该 case 的日志文件夹路径
            if case_root:
                meta["log_dir"] = case_root
            subtasks_doc = None
            st_path = os.path.join(case_root, "subtasks_classified.json")
            if not os.path.exists(st_path):
                st_path = os.path.join(case_root, "subtasks.json")
            try:
                with open(st_path, "r", encoding="utf-8") as f:
                    subtasks_doc = json.load(f)
            except Exception:
                subtasks_doc = {"subtasks": []}

            subs = subtasks_doc.get("subtasks") or []
            # 计算前缀 token：第一个 subtask 的第一个 ActionEvent 的 prompt_tokens
            prefix_tokens = _compute_prefix_tokens_first_action(case, subs, idx_to_step)
            subtask_items: List[dict] = []

            for st in subs:
                ids_raw = st.get("step_ids") or []
                step_ids: List[int] = []
                for x in ids_raw:
                    try:
                        step_ids.append(int(x))
                    except Exception:
                        continue

                turns = _pair_turns_in_subtask(step_ids, idx_to_step, idx_to_completion, case)

                subtask_items.append(
                    {
                        "title": st.get("title"),
                        "summary": st.get("summary"),
                        "type": st.get("type"),  # 可能为 None（未分类）
                        "step_ids": step_ids,
                        "num_turns": len([t for t in turns if t.get("action_index") is not None]),
                        "turns": turns,
                    }
                )

            sample = {
                "case_id": meta.get("instance_id") or os.path.basename(case_root) or None,
                "metadata": meta,
                "initial_prompt": initial_prompt,
                "prefix_tokens": prefix_tokens,
                "subtasks": subtask_items,
            }
            outf.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构造包含元数据与多轮交互的子任务数据集（JSONL）")
    p.add_argument("x", help="_tmp 下的目录名，例如 parallel_2026-04-08_00-16-16")
    p.add_argument("--tmp-root", default="_tmp", help="_tmp 根目录（默认: _tmp）")
    p.add_argument("--out-dir", default=None, help="输出目录（默认: _tmp/{x}/dataset）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = build_dataset(args.tmp_root, args.x, args.out_dir)
    print(f"[DATASET] 写入: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

"""
用法示例：

1) 构造数据集到默认位置：
   python example/cost_estimation/build_dataset.py parallel_2026-04-12_10-37-20_kimi_64cases

2) 指定 _tmp 根目录：
   python example/cost_estimation/build_dataset.py parallel_2026-04-08_00-16-16 --tmp-root _tmp

3) 指定输出目录：
   python example/cost_estimation/build_dataset.py parallel_2026-04-08_00-16-16 --out-dir _tmp/parallel_2026-04-08_00-16-16/my_dataset
"""
