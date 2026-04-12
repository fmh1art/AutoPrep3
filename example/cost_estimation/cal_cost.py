from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import glob as _glob
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error
from pathlib import Path
import math


#
# 说明：以下结构示例摘自 cal_cost.ipynb 中的一个输出（已裁剪），
# 用于帮助理解并行运行日志的文件内容与层级：
#
# cases[0] ≈ {
#   'execution_steps': {
#     'initial_prompt': '...\n(Read plan at /workspace/.agents_tmp/PLAN.md ...)\n',
#     'steps': [
#       {'index': 0, 'event_type': 'ConversationStateUpdateEvent', 'role': 'environment', ...},
#       {'index': 5, 'event_type': 'ActionEvent', 'action': {'tool': 'file_editor', 'args': {...}}, 'metrics': {...}},
#       {'index': 6, 'event_type': 'ObservationEvent', 'observation': "Here's the result of running ..."},
#       ...
#     ]
#   },
#   'token_usage': {
#     'planner': {
#       'accumulated_token_usage': { 'prompt_tokens': ..., 'completion_tokens': ..., 'reasoning_tokens': ... },
#       'token_usages': [...],
#       'response_latencies': [...],
#     },
#     'execution': {
#       'accumulated_token_usage': {...},
#       'token_usages': [...],
#       'response_latencies': [...],
#     },
#     'execution_per_step': [
#       { 'response_id': '...', 'step_index': 5, 'prompt_tokens': ..., 'completion_tokens': ... },
#       ...
#     ],
#     'total': {
#       'accumulated_token_usage': { 'prompt_tokens': ..., 'completion_tokens': ..., 'reasoning_tokens': ... }
#     }
#   }
# }
#

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] 读取 JSON 失败: {path} -> {e}", file=sys.stderr)
        return None


def _load_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] 读取文本失败: {path} -> {e}", file=sys.stderr)
        return None


def load_logs(logs_dir: str) -> Dict[str, Any]:
    """从指定 logs_dir 加载日志产物。

    两种目录形态均支持：
    1) 扁平聚合形态（OpenHands 本地跑出的 token_usage.json/results.json 等）
       - 返回形如 {"mode": "aggregate", ...}
    2) SWE 评测形态（log/ 下按 case 拆分子目录，每个子目录包含 *_result.json、*_trajectory.md 等）
       - 返回形如 {"mode": "cases", "cases": [...]}，默认加载全部 case
    """

    # 尝试聚合形态
    aggregate_payload = {
        "token_usage": _load_json(os.path.join(logs_dir, "token_usage.json")),
        "results": _load_json(os.path.join(logs_dir, "results.json")),
        "execution_steps": _load_json(os.path.join(logs_dir, "execution_steps.json")),
        "planner_trajectory_md": _load_text(os.path.join(logs_dir, "planner_trajectory.md")),
        "execution_trajectory_md": _load_text(os.path.join(logs_dir, "execution_trajectory.md")),
        "log_md": _load_text(os.path.join(logs_dir, "log.md")),
    }

    if any(v is not None for v in aggregate_payload.values()):
        aggregate_payload["mode"] = "aggregate"
        return aggregate_payload

    # 回退到 case 形态：扫描所有子目录
    cases: list[Dict[str, Any]] = []
    try:
        for name in sorted(os.listdir(logs_dir)):
            case_dir = os.path.join(logs_dir, name)
            if not os.path.isdir(case_dir):
                continue
            # 典型文件
            result_json = None
            traj_md = None
            # 任取第一个匹配的 *_result.json / *_trajectory.md
            result_candidates = _glob.glob(os.path.join(case_dir, "*_result.json"))
            traj_candidates = _glob.glob(os.path.join(case_dir, "*_trajectory.md"))
            if result_candidates:
                result_json = _load_json(result_candidates[0])
            if traj_candidates:
                traj_md = _load_text(traj_candidates[0])

            summary_md = _load_text(os.path.join(case_dir, "summary.md"))
            instance_log = _load_text(os.path.join(case_dir, "instance_log.ansi"))

            cases.append(
                {
                    "case_id": name,
                    "result_json": result_json,
                    "trajectory_md": traj_md,
                    "summary_md": summary_md,
                    "instance_log_ansi": instance_log,
                    "_paths": {
                        "result": result_candidates[0] if result_candidates else None,
                        "trajectory": traj_candidates[0] if traj_candidates else None,
                        "summary": os.path.join(case_dir, "summary.md"),
                        "instance_log": os.path.join(case_dir, "instance_log.ansi"),
                    },
                }
            )
    except FileNotFoundError:
        pass

    return {"mode": "cases", "cases": cases}


def load_parallel_cases(tmp_root: str, tmp_subdir: str) -> list[Dict[str, Any]]:
    """加载并行运行的日志结构（ipynb 中的 `load_one_log` 脚本化版本）。

    目录布局示例：
    - {tmp_root}/{tmp_subdir}/log/{UID}/log/execution_steps.json
    - {tmp_root}/{tmp_subdir}/log/{UID}/log/token_usage.json
    - {tmp_root}/{tmp_subdir}/log/{UID}/*_result.json

    返回：[{ 'execution_steps': {...}, 'token_usage': {...}, 'result': {...} }, ...]
    """
    log_root = os.path.join(tmp_root, tmp_subdir, "log")
    cases: list[Dict[str, Any]] = []
    try:
        log_dirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    except FileNotFoundError:
        return cases

    for uid in sorted(log_dirs):
        # 兼容两种目录：{uid}/log 和 {uid}
        candidate1 = os.path.join(log_root, uid, "log")
        candidate2 = os.path.join(log_root, uid)
        # 选择真实存在文件的目录
        def _pick_root() -> str:
            for root in (candidate1, candidate2):
                es = os.path.join(root, "execution_steps.json")
                tu = os.path.join(root, "token_usage.json")
                if os.path.exists(es) or os.path.exists(tu):
                    return root
            # 默认回退 {uid}/log（后续使用时会判空）
            return candidate1

        final_root = _pick_root()
        es_path = os.path.join(final_root, "execution_steps.json")
        tu_path = os.path.join(final_root, "token_usage.json")
        execution_steps = _load_json(es_path) if os.path.exists(es_path) else None
        token_usage = _load_json(tu_path) if os.path.exists(tu_path) else None
        
        # 查找并加载 *_result.json
        result = None
        result_path = None
        uid_dir = os.path.join(log_root, uid)
        try:
            for filename in os.listdir(uid_dir):
                if filename.endswith("_result.json"):
                    result_path = os.path.join(uid_dir, filename)
                    result = _load_json(result_path)
                    break
        except Exception:
            pass
        
        cases.append({
            "execution_steps": execution_steps,
            "token_usage": token_usage,
            "result": result,
            "_paths": {
                "execution_steps": es_path if os.path.exists(es_path) else None,
                "token_usage": tu_path if os.path.exists(tu_path) else None,
                "result": result_path,
                "case_root": final_root,
            }
        })

    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="任务分析工具：\n"
                    "1) 从并行日志生成 subtasks 并批量分类（写回各 case 目录）；\n"
                    "2) （可选）分析 CE 结果，对比 Ground Truth，输出统计与可视化",
    )
    parser.add_argument(
        "x",
        help="_tmp 下的目录名，例如 parallel_2026-04-07_21-29-47",
    )
    parser.add_argument(
        "--tmp-root",
        default="_tmp",
        help="_tmp 根目录（默认: _tmp）",
    )
    parser.add_argument(
        "--ce-dir",
        default=None,
        help="CE 结果目录（例如 _tmp/parallel_2026-04-08_19-07-48/ce_result）。若提供，则执行 CE 对比分析",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.ce_dir:
        args.ce_dir = os.path.join(args.tmp_root, args.x, "ce_result")

    cases = load_parallel_cases(args.tmp_root, args.x)
    print(f"Loaded cases: {len(cases)} from {os.path.join(args.tmp_root, args.x, 'log')}")

    # 功能 1：子任务生成 + 批量分类
    cfg = _load_doubao_cfg(os.path.join("_config", "doubao.yaml"))
    if not cfg:
        print("[ERROR] 无法读取 _config/doubao.yaml 或缺少必要字段 (llm_name/key/openai_base_url)", file=sys.stderr)
        return 1

    # 1.1 仅为尚未生成 subtasks 标题的案例生成 subtasks
    cases_to_gen: list[dict] = []
    for case in cases:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_root = os.path.dirname(es_path)
        st_path = os.path.join(case_root, "subtasks.json")
        need_gen = True
        try:
            if os.path.exists(st_path):
                with open(st_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                subs = (doc or {}).get("subtasks") or []
                # 若已存在非空 title，则认为已经生成过，跳过
                if any((s.get("title") or "").strip() for s in subs):
                    need_gen = False
        except Exception:
            need_gen = True
        if need_gen:
            cases_to_gen.append(case)

    if cases_to_gen:
        print(f"[INFO] 需要生成 subtasks 的案例数量: {len(cases_to_gen)}")
        _ensure_subtasks(cases_to_gen, cfg, max_subtasks=6, workers=8)
    else:
        print("[INFO] 检测到所有案例均已存在 subtasks 标题，跳过生成。")

    # 1.2 仅为尚未存在分类文件的案例进行分类
    cases_to_cls: list[dict] = []
    for case in cases:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_root = os.path.dirname(es_path)
        cls_path = os.path.join(case_root, "subtasks_classified.json")
        need_cls = True
        try:
            if os.path.exists(cls_path):
                with open(cls_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                subs = (doc or {}).get("subtasks") or []
                # 若已存在非空 type，则认为已经分类过，跳过
                if any((s.get("type") or "").strip() for s in subs):
                    need_cls = False
        except Exception:
            need_cls = True
        if need_cls:
            cases_to_cls.append(case)

    if cases_to_cls:
        print(f"[INFO] 需要进行分类的案例数量: {len(cases_to_cls)}")
        taxonomy, assignments = _classify_all_subtasks_batched(cases_to_cls, cfg, batch_size=25)
        _write_classification_per_case(cases_to_cls, assignments)
        print(f"\n[SUMMARY] New Categories ({len(taxonomy)}): {sorted(taxonomy)}")
    else:
        print("[INFO] 检测到所有案例均已存在 subtasks 分类，跳过分类。")

    # 功能 2：CE 结果对比分析（可选）
    if args.ce_dir:
        try:
            _analyze_ce_results(args.ce_dir)
        except Exception as e:
            print(f"[WARN] CE 分析失败: {e}", file=sys.stderr)
    return 0


# -------------------------------
# LLM 子任务划分实现
# -------------------------------

_CFG_LOCK = threading.Lock()


def _load_doubao_cfg(path: str) -> dict | None:
    """轻量 YAML 解析：读取 _config/doubao.yaml。

    期望字段：llm_name, key, openai_base_url
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
        cfg: dict[str, str] = {}
        for ln in lines:
            if ":" in ln:
                k, v = ln.split(":", 1)
                cfg[k.strip()] = v.strip()
        if all(k in cfg for k in ("llm_name", "key", "openai_base_url")):
            return cfg
        return None
    except Exception as e:
        print(f"[WARN] 读取配置失败: {path} -> {e}", file=sys.stderr)
        return None


def _build_steps_brief(steps: list[dict], max_chars: int = 5000) -> str:
    """将 execution_steps 的 steps 列表压缩为简要文本，控制提示长度。"""
    lines: list[str] = []
    for s in steps:
        idx = s.get("index")
        et = s.get("event_type")
        role = s.get("role")
        action = s.get("action") or {}
        obs = s.get("observation") or ""
        tool = (action or {}).get("tool") or ""
        summ = (action or {}).get("summary") or ""
        # 简要行：Index | Event | Tool | Summary | Observation(preview)
        obs_preview = str(obs).replace("\n", " ")[:200]
        line = f"[{idx}] {et} | role={role} | tool={tool} | action={summ} | obs={obs_preview}"
        lines.append(line)
        if sum(len(x) for x in lines) > max_chars:
            lines.append("... [TRUNCATED]")
            break
    return "\n".join(lines)


def _llm_chat(cfg: dict, prompt: str, timeout: float = 40.0) -> str:
    """以 OpenAI 兼容接口调用 LLM（POST /chat/completions）。

    兼容性增强：部分模型不支持 `response_format: {type: json_object}`，
    遇到 400 且错误信息包含该字段时，自动重试一次（去掉 response_format）。
    """
    url = cfg["openai_base_url"].rstrip("/") + "/chat/completions"

    def _build_body(with_json_format: bool) -> dict:
        body = {
            "model": cfg["llm_name"],
            "messages": [
                {"role": "system", "content": "You are a senior code analyst. Output strictly in JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        if with_json_format:
            body["response_format"] = {"type": "json_object"}
        return body

    def _post(body: dict) -> str:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {cfg['key']}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            return (
                payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

    # 尝试带 json_object 格式
    try:
        return _post(_build_body(with_json_format=True))
    except urllib.error.HTTPError as e:
        err_txt = e.read().decode("utf-8", errors="ignore")
        # 若是不支持 response_format 的错误，回退重试
        if e.code == 400 and ("response_format" in err_txt or "json_object is not supported" in err_txt):
            try:
                return _post(_build_body(with_json_format=False))
            except Exception as e2:
                raise RuntimeError(f"LLM fallback failed: {e2}")
        raise RuntimeError(f"LLM HTTPError {e.code}: {err_txt}")
    except Exception as e:
        # 其他错误也尝试一次不带 response_format 的重试
        try:
            return _post(_build_body(with_json_format=False))
        except Exception:
            raise RuntimeError(f"LLM request failed: {e}")


def _subtask_prompt(initial_prompt: str | None, steps: list[dict], max_subtasks: int) -> str:
    brief = _build_steps_brief(steps)
    ip = initial_prompt or "(no initial prompt)"
    return (
        "Read the following agent execution steps (sorted by index). Your task is to segment ALL steps into "
        f"non-overlapping, consecutive subtasks, with at most {max_subtasks} subtasks.\n\n"
        "Constraints:\n"
        "- Cover every step exactly once (no omissions, no overlaps).\n"
        "- Each subtask's step_ids must be a contiguous, ascending range (e.g., [1,2,3] or [10,11]).\n"
        "- Produce clear and concise ENGLISH titles and summaries for each subtask.\n\n"
        "Input:\n"
        f"Initial Prompt:\n{ip}\n\n"
        f"Steps (brief):\n{brief}\n\n"
        "Output (STRICT JSON only, no extra text):\n"
        "{\n  \"subtasks\": [\n    {\n      \"title\": \"...\",\n      \"step_ids\": [1,2,3],\n      \"summary\": \"...\"\n    }\n  ]\n}"
    )


def _gen_subtasks_for_cases_parallel(
    cases: list[Dict[str, Any]], cfg: dict, max_subtasks: int, workers: int
) -> list[dict]:
    """对并行日志中的每个案例并行生成子任务划分。"""
    outputs: list[dict] = [None] * len(cases)

    def _worker(i: int, case: dict) -> None:
        es = (case or {}).get("execution_steps") or {}
        steps = es.get("steps") or []
        ip = es.get("initial_prompt")
        if not steps:
            outputs[i] = {"subtasks": [], "_error": "no_steps"}
            return
        prompt = _subtask_prompt(ip, steps, max_subtasks)
        try:
            text = _llm_chat(cfg, prompt)
            # 解析 JSON
            result = None
            try:
                result = json.loads(text)
            except Exception:
                # 有些模型会把 JSON 包裹在代码块，粗暴清洗
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("` ")
                result = json.loads(cleaned)
            outputs[i] = result
        except Exception as e:
            outputs[i] = {"subtasks": [], "_error": str(e)}

    with ThreadPoolExecutor(max_workers=max(1, int(workers)) ) as ex:
        futs = [ex.submit(_worker, i, c) for i, c in enumerate(cases)]
        for _ in as_completed(futs):
            pass
    return outputs


# -------------------------------
# 相关性分析实现
# -------------------------------

def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    import re
    # 简易估计：单词 + 标点片段计数
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _infer_subtask_type(title: str, summary: str, steps_slice: list[dict]) -> str:
    t = f"{(title or '').lower()} {(summary or '').lower()}"
    if any(k in t for k in ["install", "pip", "conda", "build", "compile"]):
        return "install/build"
    if any(k in t for k in ["test", "pytest", "unit test", "verify", "assert"]):
        return "test/verify"
    if any(k in t for k in ["edit", "modify", "implement", "refactor", "create file", "replace", "patch"]):
        return "code/edit"
    if any(k in t for k in ["read", "view", "inspect", "search", "find", "explore"]):
        return "read/explore"
    # 回退：根据工具占比判断
    tool_counts: dict[str, int] = {}
    for s in steps_slice:
        tool = ((s.get("action") or {}).get("tool") or "").lower()
        if not tool:
            continue
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    if tool_counts:
        top_tool = max(tool_counts.items(), key=lambda x: x[1])[0]
        if "terminal" in top_tool:
            return "run/terminal"
        if "file_editor" in top_tool:
            return "read/edit"
    return "other"


def _collect_step_mappings(case: dict) -> tuple[dict, dict, dict]:
    """返回 (index->step payload, index->completion_tokens, index->observation_text)。"""
    es = (case or {}).get("execution_steps") or {}
    steps: list[dict] = es.get("steps") or []
    idx_to_step = {int(s.get("index")): s for s in steps if "index" in s}

    tu = (case or {}).get("token_usage") or {}
    exec_block = (tu.get("execution") or {})
    per_step = tu.get("execution_per_step")

    idx_to_completion: dict[int, int] = {}
    if isinstance(per_step, list) and per_step:
        for item in per_step:
            try:
                idx = int(item.get("step_index")) if item.get("step_index") is not None else None
                if idx is None:
                    continue
                idx_to_completion[idx] = int(item.get("completion_tokens", 0) or 0)
            except Exception:
                continue
    else:
        # 回退：通过 response_id 映射
        resp_to_idx: dict[str, int] = {}
        for s in steps:
            if s.get("event_type") == "ActionEvent":
                rid = (s.get("llm_response_id") or s.get("action", {}).get("response_id") or "")
                if rid:
                    resp_to_idx[str(rid)] = int(s.get("index"))
        for u in (exec_block.get("token_usages") or []):
            rid = str(u.get("response_id", ""))
            if rid and rid in resp_to_idx:
                idx_to_completion[resp_to_idx[rid]] = int(u.get("completion_tokens", 0) or 0)

    idx_to_observation: dict[int, str] = {}
    for s in steps:
        if s.get("event_type") == "ObservationEvent":
            idx_to_observation[int(s.get("index"))] = s.get("observation") or ""

    return idx_to_step, idx_to_completion, idx_to_observation


def _analyze_correlation(cases: list[dict]) -> tuple[dict, dict]:
    output_per_type, obs_per_type = _collect_tokens_by_type(cases)

    def _summarize(d: dict[str, list[int]]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for t, arr in d.items():
            arr2 = [int(x) for x in arr]
            if not arr2:
                continue
            arr2.sort()
            n = len(arr2)
            mean = sum(arr2) / n
            median = arr2[n // 2] if n % 2 == 1 else (arr2[n//2 - 1] + arr2[n//2]) / 2
            out[t] = {"count": n, "mean": mean, "median": median}
        return out

    return _summarize(output_per_type), _summarize(obs_per_type)


def _collect_tokens_by_type(cases: list[dict]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    output_per_type: dict[str, list[int]] = {}
    obs_per_type: dict[str, list[int]] = {}

    for case in cases:
        idx_to_step, idx_to_completion, idx_to_obs = _collect_step_mappings(case)

        # 读取 subtasks
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_dir = os.path.dirname(es_path)
        # 优先使用 LLM 分类后的文件；若不存在则退回原 subtasks.json + 规则型推断
        classified_path = os.path.join(case_dir, "subtasks_classified.json")
        subtasks_path = os.path.join(case_dir, "subtasks.json")
        subtasks = []
        use_llm_type = False
        try:
            if os.path.exists(classified_path):
                with open(classified_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                subtasks = doc.get("subtasks") or []
                use_llm_type = True
            else:
                with open(subtasks_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                subtasks = doc.get("subtasks") or []
        except Exception:
            subtasks = []
        for st in subtasks:
            ids = st.get("step_ids") or []
            t = (st.get("type") or "").strip()
            if use_llm_type and t:
                st_type = t
            else:
                title = st.get("title") or ""
                summary = st.get("summary") or ""
                steps_slice = [idx_to_step.get(int(i)) for i in ids if isinstance(i, int) or (isinstance(i, str) and i.isdigit())]
                steps_slice = [s for s in steps_slice if s]
                st_type = _infer_subtask_type(title, summary, steps_slice)

            for i in ids:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if idx_to_step.get(ii, {}).get("event_type") == "ActionEvent":
                    output_per_type.setdefault(st_type, []).append(int(idx_to_completion.get(ii, 0)))
                elif idx_to_step.get(ii, {}).get("event_type") == "ObservationEvent":
                    obs_txt = idx_to_obs.get(ii, "")
                    obs_per_type.setdefault(st_type, []).append(_approx_token_count(obs_txt))

    return output_per_type, obs_per_type


def _plot_boxplot(data: dict[str, list[int]], title: str, ylabel: str, output_path: str) -> bool:
    # 延迟导入，避免环境无 matplotlib 时报错影响其他功能
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    filtered = {k: [int(x) for x in v if x is not None] for k, v in data.items() if v}
    if not filtered:
        return False

    labels = sorted(filtered.keys())
    values = [filtered[k] for k in labels]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=160)
    finally:
        plt.close()
    return True


# -------------------------------
# CE 结果分析（估计 vs 真实）
# -------------------------------

def _load_ce_payloads(ce_dir: str) -> List[dict]:
    """加载 ce_result 目录下的 <instance_id>.json 列表，忽略 index.json。"""
    out: List[dict] = []
    try:
        for name in sorted(os.listdir(ce_dir)):
            if not name.endswith(".json"):
                continue
            if name == "index.json":
                continue
            p = os.path.join(ce_dir, name)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return out


def _safe_div(a: float, b: float) -> float:
    try:
        return a / b if b else 0.0
    except Exception:
        return 0.0


def _analyze_ce_results(ce_dir: str) -> None:
    """对比 CE 估计与 Ground Truth，输出 CSV 与图表到 ce_dir。"""
    payloads = _load_ce_payloads(ce_dir)
    if not payloads:
        print(f"[WARN] 未在 {ce_dir} 找到 CE 结果文件")
        return

    rows: List[dict] = []
    err_out_tokens: List[float] = []
    err_obs_tokens: List[float] = []
    err_steps: List[float] = []
    gt_out_all: List[int] = []
    pred_out_all: List[int] = []
    gt_obs_all: List[int] = []
    pred_obs_all: List[int] = []

    for item in payloads:
        instance_id = item.get("instance_id")
        ce_results: List[dict] = item.get("ce_results", [])
        gt_subs: List[dict] = (item.get("ground_truth") or {}).get("subtasks", [])
        gt_map = {i + 1: st for i, st in enumerate(gt_subs)}

        for ce in ce_results:
            idx = int(ce.get("subtask_non_negative_idx", 0) or 0)
            gt = gt_map.get(idx)
            if not gt:
                continue

            pred_steps = int(ce.get("taken_steps", 0) or 0)
            gt_steps = len(gt.get("tool_list", []) or [])
            step_err = abs(pred_steps - gt_steps)

            pred_out_sum = sum(int(x) for x in (ce.get("output_token_list") or []))
            gt_out_sum = sum(int(x) for x in (gt.get("out_token_list") or []))
            pred_obs_sum = sum(int(x) for x in (ce.get("observation_token_list") or []))
            gt_obs_sum = sum(int(x) for x in (gt.get("obs_token_list") or []))

            # 工具集合 Jaccard
            pred_tools = set(str(t).strip() for t in (ce.get("taken_tool_list") or []))
            gt_tools = set(str(t).strip() for t in (gt.get("tool_list") or []))
            inter = len(pred_tools & gt_tools)
            uni = len(pred_tools | gt_tools)
            jacc = _safe_div(inter, uni)

            rows.append(
                {
                    "instance_id": instance_id,
                    "subtask_idx": idx,
                    "title": ce.get("title") or gt.get("title") or "",
                    "pred_steps": pred_steps,
                    "gt_steps": gt_steps,
                    "abs_step_err": step_err,
                    "pred_out_tokens": pred_out_sum,
                    "gt_out_tokens": gt_out_sum,
                    "pred_obs_tokens": pred_obs_sum,
                    "gt_obs_tokens": gt_obs_sum,
                    "tool_jaccard": jacc,
                }
            )

            err_steps.append(step_err)
            err_out_tokens.append(abs(pred_out_sum - gt_out_sum))
            err_obs_tokens.append(abs(pred_obs_sum - gt_obs_sum))
            gt_out_all.append(gt_out_sum)
            pred_out_all.append(pred_out_sum)
            gt_obs_all.append(gt_obs_sum)
            pred_obs_all.append(pred_obs_sum)

    # 写出 CSV
    out_csv = os.path.join(ce_dir, "ce_vs_gt_summary.csv")
    try:
        import csv
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "instance_id",
                    "subtask_idx",
                    "title",
                    "pred_steps",
                    "gt_steps",
                    "abs_step_err",
                    "pred_out_tokens",
                    "gt_out_tokens",
                    "pred_obs_tokens",
                    "gt_obs_tokens",
                    "tool_jaccard",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[WRITE] {out_csv}")
    except Exception as e:
        print(f"[WARN] 写入 CSV 失败: {e}", file=sys.stderr)

    # 统计指标
    def _mae(arr: List[float]) -> float:
        return float(sum(arr)) / len(arr) if arr else 0.0

    mae_steps = _mae(err_steps)
    mae_out = _mae(err_out_tokens)
    mae_obs = _mae(err_obs_tokens)
    print(f"[CE] MAE steps={mae_steps:.3f}, out_tokens={mae_out:.3f}, obs_tokens={mae_obs:.3f}")

    # 可视化
    def _scatter(x: List[int], y: List[int], title: str, xlab: str, ylab: str, out_path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return False
        if not x or not y or len(x) != len(y):
            return False
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, s=10, alpha=0.6)
        m = max(max(x), max(y)) if x and y else 1
        plt.plot([0, m], [0, m], 'r--', linewidth=1)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.tight_layout()
        try:
            plt.savefig(out_path, dpi=160)
        finally:
            plt.close()
        return True

    _scatter(gt_out_all, pred_out_all, "CE vs GT (Output Tokens)", "GT out tokens", "Pred out tokens", os.path.join(ce_dir, "scatter_out_tokens.png"))
    _scatter(gt_obs_all, pred_obs_all, "CE vs GT (Observation Tokens)", "GT obs tokens", "Pred obs tokens", os.path.join(ce_dir, "scatter_obs_tokens.png"))

    # 步数误差直方图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.hist(err_steps, bins=range(0, int(max(err_steps)+2)) if err_steps else 10, edgecolor='black')
        plt.title("Absolute Step Error Histogram")
        plt.xlabel("|pred_steps - gt_steps|")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(ce_dir, "hist_abs_step_error.png"), dpi=160)
        plt.close()
    except Exception:
        pass


# -------------------------------
# Subtask 列表与 LLM 分类
# -------------------------------

def _print_all_subtasks(cases: list[dict]) -> None:
    for case in cases:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_root = os.path.dirname(es_path)
        subtasks_path = os.path.join(case_root, "subtasks.json")
        try:
            with open(subtasks_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {"subtasks": []}
        subs = doc.get("subtasks") or []
        print(f"\n[CASE] {case_root} | subtasks={len(subs)}")
        for i, st in enumerate(subs):
            title = st.get("title") or ""
            summary = st.get("summary") or ""
            ids = st.get("step_ids") or []
            print(f"- subtask[{i}] title='{title}' steps={len(ids)}")
            print(f"  summary: {summary}")


def _classify_subtasks_for_cases(cases: list[dict], cfg: dict, taxonomy: list[str], workers: int) -> None:
    # 并行对每个 subtask 分类，写回 subtasks_classified.json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _case_worker(case: dict) -> tuple[str, list[dict]]:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            return ("", [])
        case_root = os.path.dirname(es_path)
        # 加载 subtasks
        try:
            with open(os.path.join(case_root, "subtasks.json"), "r", encoding="utf-8") as f:
                subs_doc = json.load(f)
        except Exception:
            subs_doc = {"subtasks": []}
        subs = subs_doc.get("subtasks") or []
        # 加载 steps 简述
        es = (case or {}).get("execution_steps") or {}
        steps = es.get("steps") or []
        idx_map = {int(s.get("index")): s for s in steps if "index" in s}
        results: list[dict] = []
        for st in subs:
            ids = st.get("step_ids") or []
            sl = [idx_map.get(int(i)) for i in ids if (isinstance(i, int) or (isinstance(i, str) and i.isdigit()))]
            sl = [x for x in sl if x]
            brief = _build_steps_brief(sl, max_chars=3000)
            prompt = _classification_prompt(taxonomy, st.get("title"), st.get("summary"), brief)
            try:
                txt = _llm_chat(cfg, prompt, timeout=60.0)
                payload = _safe_json_parse(txt)
            except Exception as e:
                payload = {"type": "Unknown", "confidence": 0.0, "reason": str(e)}
            results.append({
                "title": st.get("title"),
                "summary": st.get("summary"),
                "step_ids": ids,
                "classification": payload,
            })
        return (case_root, results)

    outs: list[tuple[str, list[dict]]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_case_worker, c) for c in cases]
        for f in as_completed(futs):
            outs.append(f.result())

    for case_root, res in outs:
        if not case_root:
            continue
        out_path = os.path.join(case_root, "subtasks_classified.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"subtasks": res}, f, ensure_ascii=False, indent=2)
            print(f"[WRITE] {out_path}")
        except Exception as e:
            print(f"[WARN] 写入分类文件失败: {out_path} -> {e}", file=sys.stderr)


def _classification_prompt(taxonomy: list[str], title: str | None, summary: str | None, steps_brief: str) -> str:
    types = "\n".join(f"- {t}" for t in taxonomy)
    t = title or "(no title)"
    s = summary or "(no summary)"
    return (
        "You are an expert software agent analyst. Classify the following subtask into ONE of the predefined categories.\n\n"
        "Categories:\n" + types + "\n\n"
        "Subtask:\n"
        f"Title: {t}\n"
        f"Summary: {s}\n\n"
        "Steps (brief):\n" + steps_brief + "\n\n"
        "Output strictly JSON (no extra text):\n"
        "{\n  \"type\": \"<one of the categories>\",\n  \"confidence\": 0.0_to_1.0,\n  \"reason\": \"short rationale\"\n}"
    )


def _safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("` ")
        try:
            return json.loads(cleaned)
        except Exception:
            return {"type": "Unknown", "confidence": 0.0, "reason": "parse_error"}


# -------------------------------
# 一键流程：生成 subtasks + 批量分类（新增功能）
# -------------------------------

def _ensure_subtasks(cases: list[dict], cfg: dict, max_subtasks: int = 6, workers: int = 8) -> None:
    """仅为缺失/空标题的案例生成并写入 subtasks.json（存在有效标题则跳过）。"""
    # 过滤出需要生成的案例
    to_process: list[Tuple[int, dict, str]] = []  # (index, case, final_root)
    for idx, case in enumerate(cases):
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if es_path:
            final_root = os.path.dirname(es_path)
        else:
            final_root = paths.get("case_root") or ""
        if not final_root:
            continue
        os.makedirs(final_root, exist_ok=True)
        st_path = os.path.join(final_root, "subtasks.json")
        need_gen = True
        try:
            if os.path.exists(st_path):
                with open(st_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                subs = (doc or {}).get("subtasks") or []
                if any((s.get("title") or "").strip() for s in subs):
                    need_gen = False
        except Exception:
            need_gen = True
        if need_gen:
            to_process.append((idx, case, final_root))

    if not to_process:
        print("[INFO] 无需生成 subtasks（均已存在有效标题）。")
        return

    # 仅对需要生成的案例调用 LLM
    cases_filtered = [c for _, c, _ in to_process]
    results = _gen_subtasks_for_cases_parallel(cases_filtered, cfg, max_subtasks=max_subtasks, workers=workers)
    for (idx, _case, final_root), item in zip(to_process, results):
        out_path = os.path.join(final_root, "subtasks.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
            print(f"[WRITE] {out_path}")
        except Exception as e:
            print(f"[WARN] 写入子任务文件失败: {out_path} -> {e}", file=sys.stderr)


def _collect_all_subtask_records(cases: list[dict]) -> list[dict]:
    """跨案例收集所有 subtask，打平为列表，便于批量分类。"""
    records: list[dict] = []
    for case in cases:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_root = os.path.dirname(es_path)
        try:
            with open(os.path.join(case_root, "subtasks.json"), "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {"subtasks": []}
        subs = doc.get("subtasks") or []
        es = (case or {}).get("execution_steps") or {}
        steps = es.get("steps") or []
        idx_map = {int(s.get("index")): s for s in steps if "index" in s}
        for local_idx, st in enumerate(subs):
            ids = st.get("step_ids") or []
            step_slice = [idx_map.get(int(i)) for i in ids if (isinstance(i, int) or (isinstance(i, str) and i.isdigit()))]
            step_slice = [x for x in step_slice if x]
            brief = _build_steps_brief(step_slice, max_chars=2500)
            records.append({
                "case_root": case_root,
                "local_index": local_idx,
                "title": st.get("title"),
                "summary": st.get("summary"),
                "step_ids": ids,
                "steps_brief": brief,
            })
    return records


def _classification_prompt_batch(existing_cats: list[str], items: list[dict]) -> str:
    cats = "\n".join(f"- {c}" for c in existing_cats) if existing_cats else "(none)"
    lines = [
        "You are an expert software agent analyst.",
        "Task: Classify each subtask into ONE category.",
        "- Use an existing category if it fits.",
        "- If none fits, create a new concise category and reuse it for similar items in future batches.",
        "- Keep categories stable across items.",
        "- Output JSON only.",
        "\nExisting categories:",
        cats,
        "\nItems:",
    ]
    for i, it in enumerate(items):
        lines.append(f"# {i}")
        lines.append(f"Title: {it.get('title') or '(no title)'}")
        lines.append(f"Summary: {it.get('summary') or '(no summary)'}")
        lines.append("Steps (brief):\n" + (it.get("steps_brief") or ""))
        lines.append("")
    lines.append(
        "\nOutput JSON schema strictly:\n{\n  \"assignments\": [ { \"i\": <int index>, \"type\": \"<category>\" } ],\n  \"new_categories\": [\"...\"]\n}"
    )
    return "\n".join(lines)


def _classify_all_subtasks_batched(cases: list[dict], cfg: dict, batch_size: int = 25) -> tuple[set[str], dict]:
    """分批对所有 subtask 分类，保持类别稳定；返回 (taxonomy, assignments)。"""
    items = _collect_all_subtask_records(cases)
    taxonomy: set[str] = set()
    assignments: dict[tuple[str, int], dict] = {}
    i = 0
    while i < len(items):
        batch = items[i:i+batch_size]
        prompt = _classification_prompt_batch(sorted(taxonomy), batch)
        try:
            txt = _llm_chat(cfg, prompt, timeout=90.0)
            payload = _safe_json_parse(txt)
        except Exception as e:
            payload = {"assignments": [], "new_categories": [], "_error": str(e)}

        for cat in payload.get("new_categories", []) or []:
            if isinstance(cat, str) and cat.strip():
                taxonomy.add(cat.strip())
        for a in payload.get("assignments", []) or []:
            try:
                bi = int(a.get("i"))
                cat = str(a.get("type", "")).strip()
            except Exception:
                continue
            if 0 <= bi < len(batch) and cat:
                rec = batch[bi]
                key = (rec["case_root"], rec["local_index"])
                assignments[key] = {"type": cat}
        i += batch_size
    return taxonomy, assignments


def _write_classification_per_case(cases: list[dict], assignments: dict) -> None:
    """将分类结果写回各 case 的 subtasks_classified.json。"""
    for case in cases:
        paths = (case or {}).get("_paths") or {}
        es_path = paths.get("execution_steps")
        if not es_path:
            continue
        case_root = os.path.dirname(es_path)
        try:
            with open(os.path.join(case_root, "subtasks.json"), "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            doc = {"subtasks": []}
        subs = doc.get("subtasks") or []
        out_subs = []
        for idx, st in enumerate(subs):
            cls = assignments.get((case_root, idx)) or {"type": "Unlabeled"}
            out_subs.append({
                "title": st.get("title"),
                "summary": st.get("summary"),
                "step_ids": st.get("step_ids"),
                "type": cls.get("type"),
            })
        out_path = os.path.join(case_root, "subtasks_classified.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"subtasks": out_subs}, f, ensure_ascii=False, indent=2)
            print(f"[WRITE] {out_path}")
        except Exception as e:
            print(f"[WARN] 写入分类文件失败: {out_path} -> {e}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

"""
运行示例：

1) 生成所有实例的 subtasks 并批量分类（会写回各 case 目录）：
   python example/cost_estimation/cal_cost.py parallel_2026-04-11_21-55-36 --ce-dir _tmp/parallel_2026-04-11_21-55-36/ce_result

依赖：_config/doubao.yaml 中需提供 openai 兼容配置：
   llm_name: <model>
   key: <api-key>
   openai_base_url: <base-url>
"""
