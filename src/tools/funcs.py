import os, json
import numpy as np
import pandas as pd
from datetime import date, datetime
from decimal import Decimal
from tqdm import tqdm
import uuid
from typing import Optional, Dict, Any

# Jinja2 渲染支持
try:
    from jinja2 import Environment, FileSystemLoader
except Exception:
    Environment = None
    FileSystemLoader = None

try:
    import tiktoken
except Exception:
    tiktoken = None

import re

def parse_any_string(rsp, code_type=None, hard_replace=None):
    if hard_replace != None:
        if not isinstance(hard_replace, list):
            hard_replace = [hard_replace]
        for hr in hard_replace:
            rsp = rsp.replace(hr, '')
    if code_type != None:
        rsp = rsp.replace('```'+code_type, '```')
    pattern = r"```(.*?)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    if rsp.count('```') < 2:
        return rsp
    if code_text.strip().startswith('python'):
        code_text = code_text.replace('python', '', 1).strip()
    if code_text.strip().startswith('SQL'):
        code_text = code_text.replace('SQL', '', 1).strip()
    if code_text.strip().startswith('sql'):
        code_text = code_text.replace('sql', '', 1).strip()
    if code_text.strip().startswith('neuralsql'):
        code_text = code_text.replace('neuralsql', '', 1).strip()
    if code_text.strip().startswith('NeuralSQL'):
        code_text = code_text.replace('NeuralSQL', '', 1).strip()
    if code_text.strip().startswith('neural_sql'):
        code_text = code_text.replace('neural_sql', '', 1).strip()
    return code_text


def all_filepaths_in_dir(root, endswith=None):
    file_paths = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if endswith is None or file.endswith(endswith):
                file_paths.append(os.path.join(subdir, file))
    return file_paths

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        d = json.loads(line)
        data.append(d)
    return data

def save_jsonl(path, datas):
    # if os.path.exists(path):
    #     print(f'Path: {path} already exists! Please delete it!')
    #     return
    
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print(f'saving jsonl object to {path}...')
    with open(path, 'w', encoding='utf-8') as f:
        for d in tqdm(datas):
            f.write(json.dumps(d, default=convert_nat) + '\n')



def convert_nat(o):
    if isinstance(o, pd._libs.missing.NAType):
        return None
    elif isinstance(o, (date, datetime)):
        return o.strftime('%Y-%m-%d %H:%M:%S') if isinstance(o, datetime) else o.strftime('%Y-%m-%d')
    elif isinstance(o, Decimal):
        return float(o)
    elif isinstance(o, uuid.UUID):
        return str(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def open_json(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(a, fn):

    dir_path = os.path.dirname(fn)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    try:
        b = json.dumps(a, default=convert_nat)
        with open(fn, 'w') as f2:
            f2.write(b)
    except Exception as e:
        print(f"Error saving JSON: {e}")


def cal_token(text: str, model: str = "cl100k_base") -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


# ---------------------------
# Jinja2 模板渲染通用工具
# ---------------------------

def get_templates_env(base_dir: Optional[str] = None):
    """获取 Jinja2 Environment。

    - 默认模板目录为相对本文件 `../prompts`（即 `src/prompts`）。
    - 如需自定义目录，可传入 `base_dir`。
    """
    if Environment is None or FileSystemLoader is None:
        raise RuntimeError("jinja2 未安装，无法进行模板渲染。请先安装 jinja2。")

    base = (
        base_dir
        if base_dir is not None
        else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
    )
    return Environment(loader=FileSystemLoader(base), autoescape=False)


def render_j2(template_name: str, context: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None) -> str:
    """渲染指定的 Jinja2 模板。

    参数：
    - template_name: 模板文件名，如 `code_agent_plan_mode_planning.j2`
    - context: 渲染上下文 dict
    - base_dir: 模板目录，可选；不传则默认 `src/prompts`
    """
    env = get_templates_env(base_dir)
    tpl = env.get_template(template_name)
    return tpl.render(**(context or {}))


def calculate_multi_step_cost_without_prefix(token_lis: list, inp_price: int, out_price: int, cache_price: int):
    """
    token_lis: list of dict, each with keys 'output_token' and 'observation_token'
    对应 [(Out_1, Obs_1), (Out_2, Obs_2), ..., (Out_L, Obs_L)]
    
    每一步成本三部分：
    - cached input  : prefix_token * cache_price
    - uncached obs  : prev_obs * inp_price  (上一步的 observation 作为当前输入)
    - output        : out * out_price
    """
    total_cost = 0.0
    prefix_token = 0
    prev_obs = 0  # 第1步没有前置 observation

    for token_metric in token_lis:
        out = token_metric['output_token']
        obs = token_metric['observation_token']

        # 三部分成本
        cached_input_cost  = prefix_token * cache_price
        uncached_obs_cost  = prev_obs * inp_price
        output_cost        = out * out_price

        total_cost += cached_input_cost + uncached_obs_cost + output_cost

        # 更新 prefix：当前轮的 out + obs 加入缓存
        prefix_token += out + obs
        # 下一轮的 uncached obs 是当前轮的 obs
        prev_obs = obs

    return total_cost


def _parse_tool_args_for_display(tool_args):
    if isinstance(tool_args, dict):
        if "_raw" in tool_args:
            try:
                parsed = json.loads(tool_args["_raw"])
                return {k: v for k, v in parsed.items() if k != "_trajectory_step_index"}
            except (json.JSONDecodeError, TypeError):
                pass
        return {k: v for k, v in tool_args.items() if k != "_trajectory_step_index"}
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
            return {k: v for k, v in parsed.items() if k != "_trajectory_step_index"}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": tool_args}
    return {}


def render_trajectory_md(trajectory: list[dict], extra_info: dict | None = None) -> str:
    lines: list[str] = []
    lines.append("# Agent Trajectory\n")

    if extra_info:
        for k, v in extra_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    for record in trajectory:
        idx = record.get("index", "?")
        role = record.get("role", "?")

        if role == "initial_prompt":
            lines.append("## Initial Prompt\n")
            msgs = record.get("messages", [])
            for mi, m in enumerate(msgs):
                m_role = m.get("role", "?")
                m_content = m.get("content", "")
                if m_content is None:
                    m_content = ""
                m_content = str(m_content)
                tool_calls = m.get("tool_calls", [])
                tool_call_id = m.get("tool_call_id", "")

                if m_role == "system":
                    lines.append(f"### [{mi}] System\n")
                    lines.append(f"```\n{m_content}\n```\n")
                elif m_role == "user":
                    lines.append(f"### [{mi}] User\n")
                    lines.append(f"```\n{m_content}\n```\n")
                elif m_role == "assistant":
                    m_reasoning = m.get("reasoning_content", "")
                    lines.append(f"### [{mi}] Assistant\n")
                    if m_reasoning:
                        lines.append(f"<details><summary>Reasoning</summary>\n\n{m_reasoning}\n\n</details>\n")
                    if m_content:
                        lines.append(f"**Content:** {m_content}\n")
                    if tool_calls:
                        for tci, tc in enumerate(tool_calls):
                            fn = tc.get("function", {})
                            tc_name = fn.get("name", "?")
                            tc_args = fn.get("arguments", "")
                            lines.append(f"**Tool Call {tci}: `{tc_name}`**\n")
                            try:
                                args_parsed = json.loads(tc_args)
                                args_display = {k: v for k, v in args_parsed.items() if k != "_trajectory_step_index"}
                                lines.append(f"```json\n{json.dumps(args_display, ensure_ascii=False, indent=2)}\n```\n")
                            except (json.JSONDecodeError, TypeError):
                                lines.append(f"```json\n{tc_args}\n```\n")
                elif m_role == "tool":
                    lines.append(f"### [{mi}] Tool Response (id={tool_call_id})\n")
                    lines.append(f"```\n{m_content}\n```\n")
                else:
                    lines.append(f"### [{mi}] {m_role}\n")
                    lines.append(f"```\n{m_content}\n```\n")
            continue

        if role == "assistant":
            thinking = record.get("thinking", "")
            reasoning = record.get("reasoning", "")
            lines.append(f"## Step {idx} — Assistant\n")
            if reasoning:
                lines.append(f"<details><summary>Reasoning</summary>\n\n{reasoning}\n\n</details>\n")
            if thinking:
                lines.append(f"**Thinking:** {thinking}\n")
            continue

        if role == "tool":
            tool_name = record.get("tool_name", "?")
            tool_args = record.get("tool_args", {})
            observation = record.get("observation", "")
            thinking = record.get("thinking", "")
            reasoning = record.get("reasoning", "")
            finish_msg = record.get("finish_message")
            useful_idx = record.get("useful_trajectory_indexes")

            lines.append(f"## Step {idx} — Tool: `{tool_name}`\n")

            if reasoning:
                lines.append(f"<details><summary>Reasoning</summary>\n\n{reasoning}\n\n</details>\n")
            if thinking:
                lines.append(f"**Thinking:** {thinking}\n")

            args_display = _parse_tool_args_for_display(tool_args)
            if args_display:
                lines.append("**Arguments:**\n")
                lines.append(f"```json\n{json.dumps(args_display, ensure_ascii=False, indent=2)}\n```\n")

            if observation:
                lines.append("**Observation:**\n")
                obs = str(observation)
                if len(obs) > 2000:
                    lines.append(f"<details><summary>Observation ({len(obs)} chars)</summary>\n\n```\n{obs}\n```\n\n</details>\n")
                else:
                    lines.append(f"```\n{obs}\n```\n")

            if finish_msg:
                lines.append(f"**Finish Message:** {finish_msg}\n")
            if useful_idx:
                lines.append(f"**Useful Trajectory Indexes:** {useful_idx}\n")

            continue

        if role == "transition":
            lines.append(f"## Transition — Operator {idx} → {idx + 1}\n")
            content = record.get("content", "")
            if content:
                lines.append(f"```\n{content}\n```\n")
            continue

        lines.append(f"## Step {idx} — {role}\n")
        content = record.get("content") or record.get("observation") or ""
        if content:
            lines.append(f"```\n{content}\n```\n")

    return "\n".join(lines)
