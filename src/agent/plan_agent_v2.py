"""
COAT V2 — Self-Evolving Planning Agent 框架。

COAT: Context-aware Orchestrated Agent with Tool-calling

V2 相比 V1/V0 的核心改进：
  三个模块协同工作：
  1. 执行模块：Planning Agent + Self-Evolve Code Agent
     - Execution Agent 初始只有 bash 工具 + 已有 skill（一开始为空）
     - Skill 是封装好的 bash 脚本 + py 脚本，放在 workspace/.skill/ 下
  2. 检测模块：当某个 operator 执行消耗过多步骤时，触发成本异常
     - 转入自进化模块
  3. 自进化模块：根据 trajectory 分析成本高昂原因，定义新 skill
     - Skill 由描述（放入 prompt）、bash 脚本、py 脚本组成
     - Skill 全局共享，一次定义所有 execution agent 都能使用
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent_optimized import CodeAgentOptimized, AgentResult
from src.agent.plan_agent import (
    PLAN_AGENT_TOOLS,
    TERMINATE_TOOL,
    VIEW_FILE_TOOL,
    SEARCH_BY_KEYWORD_TOOL,
    SubtaskResult,
    _filter_trajectory_records,
    _serialize_subagent_trajectory,
    _exec_view_file,
    _exec_search_by_keyword,
)
from src.agent.plan_agent_v1 import SubtaskResultV1
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COAT V2 Skill 定义
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    bash_script: str
    py_script: str
    created_at: float = 0.0
    source_trajectory_summary: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "bash_script": self.bash_script,
            "py_script": self.py_script,
            "created_at": self.created_at,
            "source_trajectory_summary": self.source_trajectory_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=d["name"],
            description=d["description"],
            bash_script=d["bash_script"],
            py_script=d["py_script"],
            created_at=d.get("created_at", 0.0),
            source_trajectory_summary=d.get("source_trajectory_summary", ""),
        )


class SkillRegistry:
    _instance: "SkillRegistry | None" = None

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: dict[str, Skill] = {}
        return cls._instance

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        logger.info(f"[SkillRegistry] Registered skill: {skill.name}")

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def deploy_to_workspace(self, workspace: "DockerWorkspace", repo_path: str = "/workspace") -> None:
        skill_dir = f"{repo_path}/.skill"
        workspace.execute_command(f"mkdir -p {skill_dir}", timeout=10)
        for skill in self._skills.values():
            bash_path = f"{skill_dir}/{skill.name}.sh"
            py_path = f"{skill_dir}/{skill.name}.py"
            escaped_bash = skill.bash_script.replace("'", "'\\''")
            escaped_py = skill.py_script.replace("'", "'\\''")
            workspace.execute_command(f"cat > {bash_path} << 'SKILL_EOF'\n{skill.bash_script}\nSKILL_EOF", timeout=10)
            workspace.execute_command(f"chmod +x {bash_path}", timeout=10)
            workspace.execute_command(f"cat > {py_path} << 'SKILL_EOF'\n{skill.py_script}\nSKILL_EOF", timeout=10)

    def build_skill_prompt_section(self) -> str:
        if not self._skills:
            return ""
        parts = ["## Available Skills\n", "You have the following skills available. Each skill is a bash script located at `/workspace/.skill/<name>.sh`. Call them via bash, e.g.: `bash /workspace/.skill/search_keyword.sh <args>`\n"]
        for skill in self._skills.values():
            parts.append(f"### {skill.name}\n{skill.description}\n")
        return "\n".join(parts)

    def save_to_file(self, path: str) -> None:
        data = {name: s.to_dict() for name, s in self._skills.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_skill_files(self, output_dir: str) -> None:
        skills_dir = os.path.join(output_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        for skill in self._skills.values():
            skill_dir = os.path.join(skills_dir, skill.name)
            os.makedirs(skill_dir, exist_ok=True)
            bash_path = os.path.join(skill_dir, f"{skill.name}.sh")
            py_path = os.path.join(skill_dir, f"{skill.name}.py")
            desc_path = os.path.join(skill_dir, "description.txt")
            with open(bash_path, "w", encoding="utf-8") as f:
                f.write(skill.bash_script)
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(skill.py_script)
            with open(desc_path, "w", encoding="utf-8") as f:
                f.write(skill.description)
            logger.info(f"[SkillRegistry] Exported skill files: {skill_dir}/")
        if self._skills:
            logger.info(f"[SkillRegistry] All skill files exported to {skills_dir}/")

    def load_from_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, d in data.items():
            self._skills[name] = Skill.from_dict(d)
        logger.info(f"[SkillRegistry] Loaded {len(data)} skills from {path}")

    def reset(self) -> None:
        self._skills.clear()


# ---------------------------------------------------------------------------
# COAT V2 Self-Evolve Code Agent — execution agent with only bash + skills
# ---------------------------------------------------------------------------

BASH_ONLY_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in the Docker workspace. Use `&&` or `;` "
            "to chain multiple commands. You can also call skill scripts "
            "located at /workspace/.skill/<name>.sh"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call when the task is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "useful_trajectory_indexes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indexes of steps that contain essential information for the next operator.",
                },
            },
            "required": [],
        },
    },
}


def _build_self_evolve_system_prompt(repo_path: str, skill_prompt: str) -> str:
    parts = [
        f"You are an autonomous coding agent running inside a Docker workspace.",
        f"Repository root: {repo_path}",
        "",
        "You have access to the following tools:",
        "  - bash: run shell commands. This is your ONLY tool.",
        "  - finish: terminate when the task is complete.",
        "",
    ]
    if skill_prompt:
        parts.append(skill_prompt)
        parts.append("")
    parts.extend([
        "IMPORTANT RULES:",
        "  - Always use absolute file paths (starting with /).",
        "  - Test your changes whenever possible.",
        "  - Do NOT ask the user for help. Work autonomously.",
        "  - If you get stuck, try a different approach.",
        "  - When the task is complete, call `finish`.",
        "",
        "WORKFLOW GUIDELINES:",
        "  - Use grep, find, cat, sed, awk and other standard unix tools to explore and modify code.",
        "  - Use python3 for complex operations (e.g., parsing JSON, running tests).",
        "  - Prefer using available skills over manual multi-step bash commands.",
    ])
    return "\n".join(parts)


class SelfEvolveCodeAgent:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )
        self._tools = [BASH_ONLY_TOOL, FINISH_TOOL]

    def _build_system_prompt(self, repo_path: str) -> str:
        registry = SkillRegistry()
        skill_prompt = registry.build_skill_prompt_section()
        return _build_self_evolve_system_prompt(repo_path, skill_prompt)

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(
                    messages=messages,
                    tools=self._tools,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[SelfEvolveCodeAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"SelfEvolveCodeAgent LLM call failed after 3 attempts") from last_exc

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_sev")
        os.makedirs(out_dir, exist_ok=True)

        registry = SkillRegistry()
        registry.deploy_to_workspace(workspace, repo_path)

        system_prompt = self._build_system_prompt(repo_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        trajectory: list[dict] = []
        usage_before = self._caller.get_total_usage()
        useful_trajectory_indexes: list[int] = []
        start_time = time.time()

        for step in range(self.max_steps):
            if self.max_time is not None and (time.time() - start_time) > self.max_time:
                logger.warning(f"[SelfEvolveCodeAgent] Timeout at step {step}")
                break

            response_msg = self._call_llm(messages)
            assistant_msg = self._response_to_dict(response_msg)
            messages.append(assistant_msg)

            self._save_llm_io(out_dir, step, messages, response_msg)

            tool_calls = response_msg.tool_calls or []
            if not tool_calls:
                remaining = self.max_steps - step - 1
                messages.append({
                    "role": "user",
                    "content": f"Please continue using the available tools. ({remaining} steps remaining)",
                })
                continue

            finished = False
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                if tool_name == "finish":
                    raw = tool_args.get("useful_trajectory_indexes", [])
                    if isinstance(raw, list):
                        useful_trajectory_indexes = [i for i in raw if isinstance(i, int)]
                    observation = "Agent finished."
                    finished = True
                elif tool_name == "bash":
                    command = tool_args.get("command", "")
                    if not command:
                        observation = "Error: empty command"
                    else:
                        timeout = float(tool_args.get("timeout", 120))
                        try:
                            r = workspace.execute_command(command, timeout=timeout)
                            if r.timeout_occurred:
                                observation = f"Error: Command timed out after {timeout}s"
                            else:
                                parts = []
                                if r.stdout:
                                    parts.append(r.stdout)
                                if r.stderr:
                                    parts.append(r.stderr)
                                output = "\n".join(parts) or "(no output)"
                                if len(output) > 6000:
                                    half = 3000
                                    output = output[:half] + f"\n\n... ({len(output) - 6000} chars truncated) ...\n\n" + output[-half:]
                                observation = output + f"\n[exit code: {r.exit_code}]"
                        except Exception as e:
                            observation = f"Error: {e}"
                else:
                    observation = f"Error: Unknown tool '{tool_name}'"

                trajectory.append({
                    "index": step,
                    "role": "tool",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "observation": observation,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"{observation}\n[Step {step}]",
                })

                if finished:
                    break

            if finished:
                break

            remaining = self.max_steps - step - 1
            if 0 < remaining <= 5:
                messages.append({
                    "role": "user",
                    "content": f"[WARNING] You have only {remaining} step(s) remaining. Call `finish` if done.",
                })
        else:
            logger.warning("[SelfEvolveCodeAgent] Reached max steps without finishing")

        usage_after = self._caller.get_total_usage()
        metrics = {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "accumulated_cost": 0.0,
        }

        return AgentResult(
            metrics=metrics,
            messages=messages,
            other_content={
                "useful_trajectory_indexes": useful_trajectory_indexes,
                "trajectory_records": trajectory,
                "terminated_tool": "finish" if useful_trajectory_indexes else "",
                "total_steps": len(trajectory),
            },
        )

    @staticmethod
    def _response_to_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant"}
        d["content"] = msg.content if msg.content else None
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content:
            d["reasoning_content"] = reasoning_content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    def _save_llm_io(self, out_dir: str, step: int, messages: list[dict], response_msg) -> None:
        llm_log_dir = os.path.join(out_dir, "llm_io")
        os.makedirs(llm_log_dir, exist_ok=True)
        md_path = os.path.join(llm_log_dir, f"step_{step:03d}.md")

        lines: list[str] = []
        lines.append(f"# Execution Agent (SelfEvolve) LLM IO — Step {step}\n")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append("---\n")
        lines.append("## Input Messages\n")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            lines.append(f"### [{i}] {role}\n")

            content = msg.get("content")
            if content:
                lines.append("```\n" + str(content) + "\n```\n")

            reasoning = msg.get("reasoning_content")
            if reasoning:
                lines.append("<details><summary>Reasoning</summary>\n\n```\n" + str(reasoning) + "\n```\n\n</details>\n")

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tci, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "?")
                    tc_args = fn.get("arguments", "")
                    lines.append(f"**Tool Call {tci}: `{tc_name}`**\n")
                    try:
                        args_parsed = json.loads(tc_args)
                        lines.append("```json\n" + json.dumps(args_parsed, ensure_ascii=False, indent=2) + "\n```\n")
                    except (json.JSONDecodeError, TypeError):
                        lines.append("```json\n" + tc_args + "\n```\n")

            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                tool_content = msg.get("content", "")
                if len(tool_content) > 3000:
                    tool_content = tool_content[:3000] + "\n... (truncated)"
                lines.append(f"**Tool Response (id={tool_call_id})**\n")
                lines.append("```\n" + tool_content + "\n```\n")

        lines.append("---\n")
        lines.append("## Output Response\n")

        resp_content = response_msg.content if response_msg.content else ""
        if resp_content:
            lines.append("### Content\n")
            lines.append("```\n" + resp_content + "\n```\n")

        resp_reasoning = getattr(response_msg, "reasoning_content", None) or ""
        if resp_reasoning:
            lines.append("<details><summary>Reasoning</summary>\n\n```\n" + resp_reasoning + "\n```\n\n</details>\n")

        resp_tool_calls = response_msg.tool_calls or []
        if resp_tool_calls:
            lines.append("### Tool Calls\n")
            for tci, tc in enumerate(resp_tool_calls):
                fn = tc.function
                tc_name = fn.name
                tc_args = fn.arguments
                lines.append(f"**{tci}. `{tc_name}`**\n")
                try:
                    args_parsed = json.loads(tc_args)
                    lines.append("```json\n" + json.dumps(args_parsed, ensure_ascii=False, indent=2) + "\n```\n")
                except (json.JSONDecodeError, TypeError):
                    lines.append("```json\n" + tc_args + "\n```\n")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to save LLM IO for step {step}: {e}")


# ---------------------------------------------------------------------------
# COAT V2 Detection Module — cost anomaly detection
# ---------------------------------------------------------------------------

@dataclass
class CostAnomaly:
    subtask_index: int
    subtask_description: str
    total_steps: int
    threshold: int
    trajectory_summary: str


class CostDetector:
    def __init__(
        self,
        step_threshold: int = 40,
        token_threshold: int = 100000,
    ):
        self.step_threshold = step_threshold
        self.token_threshold = token_threshold

    def check(
        self,
        subtask_index: int,
        subtask_description: str,
        trajectory_records: list[dict],
        metrics: dict,
    ) -> CostAnomaly | None:
        total_steps = len([r for r in trajectory_records if r.get("role") == "tool"])
        total_tokens = metrics.get("total_tokens", 0)

        if total_steps >= self.step_threshold or total_tokens >= self.token_threshold:
            summary = self._summarize_trajectory(trajectory_records)
            return CostAnomaly(
                subtask_index=subtask_index,
                subtask_description=subtask_description,
                total_steps=total_steps,
                threshold=self.step_threshold,
                trajectory_summary=summary,
            )
        return None

    @staticmethod
    def _summarize_trajectory(records: list[dict]) -> str:
        parts = []
        for rec in records:
            if rec.get("role") != "tool":
                continue
            idx = rec.get("index", "?")
            tool_name = rec.get("tool_name", "?")
            tool_args = rec.get("tool_args", {})
            observation = rec.get("observation", "")

            if tool_name == "bash":
                cmd = tool_args.get("command", "")[:200]
                obs_preview = observation[:300] if observation else ""
                parts.append(f"Step {idx} [bash]: {cmd}\n  Output: {obs_preview}")
            else:
                parts.append(f"Step {idx} [{tool_name}]")

            if len(parts) >= 50:
                parts.append(f"... ({len(records)} total steps, truncated)")
                break

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# COAT V2 Self-Evolution Module — skill generation from trajectory
# ---------------------------------------------------------------------------

SKILL_EVOLUTION_PROMPT = """You are a skill designer for a coding agent. The agent has only a `bash` tool and needs to be more efficient.

## Problem

A subtask consumed too many steps ({total_steps} steps, threshold is {threshold}). Here is the trajectory:

{trajectory_summary}

## Your Task

Analyze the trajectory and determine whether there is a **genuinely reusable, multi-step pattern** that can be encapsulated as a skill.

**IMPORTANT — Quality over quantity:**
- Only create a skill if you can identify a **clear, repetitive pattern** that would save significant steps in FUTURE tasks (not just this one).
- Do NOT create a skill for one-off operations, trivial single-command operations, or task-specific logic.
- Do NOT create a skill that merely wraps a single bash command (e.g., `grep`, `cat`, `find`) — that adds no value.
- A good skill candidate: a multi-step operation that the agent repeats across different tasks (e.g., search + filter + format, or find + edit + verify).
- When in doubt, **skip**. Creating unnecessary skills pollutes the agent's prompt and hurts performance.

## Skill Definition

A skill consists of:
1. **name**: A short, descriptive name (snake_case, e.g., `search_keyword`, `find_and_replace`)
2. **description**: A detailed description of what the skill does, its parameters, and usage. This goes into the agent's system prompt.
3. **bash_script**: A bash script that calls the python script. It should:
   - Accept command-line arguments
   - Call the python script with those arguments
   - Be located at `/workspace/.skill/<name>.sh`
4. **py_script**: A Python script that implements the skill logic. It should:
   - Accept arguments via sys.argv or argparse
   - Be located at `/workspace/.skill/<name>.py`
   - Use only standard library + common packages (os, sys, json, re, subprocess, argparse)

## Important Notes

- The scripts will be deployed to `/workspace/.skill/` in the Docker workspace
- Use absolute paths starting with `/workspace/` in the scripts
- The bash script should be executable (#!/bin/bash)
- The python script should have a proper shebang (#!/usr/bin/env python3)
- Make the skill general-purpose, not tied to this specific task
- The skill should handle errors gracefully

## Output Format

If you identify a genuinely useful, reusable pattern, respond with a JSON object:
```json
{{
  "name": "skill_name",
  "description": "Detailed description of the skill, its parameters, and how to use it. Example: search_keyword.sh <keyword> <path> [content|filename] - Searches for a keyword in files under the given path...",
  "bash_script": "#!/bin/bash\\n# Skill: skill_name\\nset -e\\npython3 /workspace/.skill/skill_name.py \"$@\"\\n",
  "py_script": "#!/usr/bin/env python3\\nimport sys\\nimport os\\n...\\n"
}}
```

If you cannot identify a genuinely useful and reusable skill, you MUST respond with:
```json
{{"skip": true, "reason": "explanation of why no skill is needed"}}
```

Common reasons to skip:
- The high step count is due to task complexity, not repetitive patterns
- The operations are one-off and task-specific, not reusable
- The existing bash commands are already efficient enough
- No clear multi-step pattern can be extracted
"""


class SkillEvolver:
    def __init__(self, llm_cfg: dict):
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    def evolve(self, anomaly: CostAnomaly) -> tuple[Skill | None, str]:
        prompt = SKILL_EVOLUTION_PROMPT.format(
            total_steps=anomaly.total_steps,
            threshold=anomaly.threshold,
            trajectory_summary=anomaly.trajectory_summary[:8000],
        )

        messages = [
            {"role": "system", "content": "You are a skill designer. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):
            try:
                response = self._caller.chat(messages=messages)
                content = response or ""
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                parsed = json.loads(content)
                if parsed.get("skip"):
                    reason = parsed.get("reason", "no reason provided")
                    logger.info(f"[SkillEvolver] Skipped: {reason}")
                    return None, reason

                skill = Skill(
                    name=parsed["name"],
                    description=parsed["description"],
                    bash_script=parsed["bash_script"],
                    py_script=parsed["py_script"],
                    source_trajectory_summary=anomaly.trajectory_summary[:500],
                )
                logger.info(f"[SkillEvolver] Generated skill: {skill.name}")
                return skill, ""
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillEvolver] Parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillEvolver] Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillEvolver] Failed to generate skill after 3 attempts")
        return None, "failed to parse LLM response after 3 attempts"


# ---------------------------------------------------------------------------
# COAT V2 SubAgentV2 — wraps SelfEvolveCodeAgent
# ---------------------------------------------------------------------------

class SubAgentV2:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self.selective_fallback_rule = selective_fallback_rule

    def run(
        self,
        original_task: str,
        subtask_description: str,
        related_trajectories: list[dict],
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        llm_choice: Literal["cheap", "expensive"] = "cheap",
    ) -> SubtaskResultV1:
        llm_cfg = self.expensive_llm_cfg if llm_choice == "expensive" else self.cheap_llm_cfg
        agent = SelfEvolveCodeAgent(
            llm_cfg=llm_cfg,
            max_steps=self.max_steps,
            max_retries_per_call=self.max_retries_per_call,
            max_time=self.max_time,
        )

        focus_prompt = render_j2("subagent_focus.j2", context={
            "original_task": original_task,
            "subtask_description": subtask_description,
            "related_trajectories": related_trajectories,
        })

        task_query = render_j2("query.j2", context={
            "problem_statement": original_task,
            "repo_path": repo_path,
        })

        instruction = task_query + "\n\n" + focus_prompt

        logger.info(
            f"[COAT V2 SubAgent] Running subtask: {subtask_description[:100]} | "
            f"llm={llm_choice} | skills={SkillRegistry().skill_names()}"
        )

        try:
            result = agent.run(
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                callbacks=callbacks,
                output_dir=output_dir,
            )

            other = result.other_content or {}
            useful_indexes = other.get("useful_trajectory_indexes", [])
            trajectory_records = other.get("trajectory_records", [])
            terminated_tool = other.get("terminated_tool", "")

            completed_normally = terminated_tool == "finish"

            if not useful_indexes and self.selective_fallback_rule != "none":
                total_steps = self._compute_total_steps(trajectory_records)
                useful_indexes = self._apply_fallback_rule(total_steps)

            hit_max_steps = len(result.messages if hasattr(result, "messages") else []) >= self.max_steps * 2

            return SubtaskResultV1(
                subtask=subtask_description,
                useful_trajectory_indexes=useful_indexes,
                messages=result.messages if hasattr(result, "messages") else [],
                trajectory_records=trajectory_records,
                metrics=result.metrics if hasattr(result, "metrics") else {},
                completed_normally=completed_normally and not hit_max_steps,
                hit_max_steps=hit_max_steps,
                llm_used=llm_choice,
            )
        except Exception as e:
            logger.error(f"[COAT V2 SubAgent] Execution failed: {e}")
            return SubtaskResultV1(
                subtask=subtask_description,
                error=str(e),
                completed_normally=False,
                llm_used=llm_choice,
            )

    @staticmethod
    def _compute_total_steps(records: list[dict]) -> int:
        max_step = -1
        for rec in records:
            if rec.get("role") == "tool":
                idx = rec.get("index", -1)
                if isinstance(idx, int) and idx > max_step:
                    max_step = idx
        return max_step + 1 if max_step >= 0 else 0

    def _apply_fallback_rule(self, total_steps: int) -> list[int]:
        if total_steps == 0:
            return []
        if self.selective_fallback_rule == "all":
            return list(range(total_steps))
        elif self.selective_fallback_rule == "last_half":
            count = max(1, total_steps // 2)
        elif self.selective_fallback_rule == "last_third":
            count = max(1, total_steps // 3)
        else:
            count = max(1, total_steps // 2)
        return list(range(total_steps - count, total_steps))


# ---------------------------------------------------------------------------
# COAT V2 PlanAgentV2 — planning agent (same as V1)
# ---------------------------------------------------------------------------

class PlanAgentV2:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_planning_steps: int = 20,
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_planning_steps = max_planning_steps
        self.llm_selection_strategy = llm_selection_strategy

        self._caller = SimpleAPICaller(
            llm_name=expensive_llm_cfg.get("llm_name", expensive_llm_cfg.get("model", "").replace("openai/", "")),
            api_key=expensive_llm_cfg.get("key", expensive_llm_cfg.get("api_key", "")),
            base_url=expensive_llm_cfg.get("openai_base_url", expensive_llm_cfg.get("base_url", None)),
            api_version=expensive_llm_cfg.get("api_version", None),
            cache_server_url=expensive_llm_cfg.get("cache_server_url"),
        )

    @property
    def caller(self):
        return self._caller

    def _build_system_prompt(self, completed_subtasks: list[dict]) -> str:
        context = {"completed_subtasks": completed_subtasks}
        if self.llm_selection_strategy == "llm_judged":
            context["llm_judged_mode"] = True
        return render_j2("plan_agent_step_by_step.j2", context=context)

    def _call_llm(self, messages: list[dict]):
        from src.agent.plan_agent_v1 import PLAN_AGENT_TOOLS_V1
        tools = PLAN_AGENT_TOOLS_V1 if self.llm_selection_strategy == "llm_judged" else PLAN_AGENT_TOOLS
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(messages=messages, tools=tools)
            except Exception as e:
                last_exc = e
                logger.warning(f"[COAT V2 PlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"COAT V2 PlanAgent LLM call failed after 3 attempts") from last_exc


# ---------------------------------------------------------------------------
# COAT V2 PlanAgentPipelineV2 — full pipeline with self-evolution
# ---------------------------------------------------------------------------

@dataclass
class PlanAgentResultV2:
    subtask_results: list[SubtaskResultV1] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)
    evolved_skills: list[str] = field(default_factory=list)
    cost_anomalies: list[dict] = field(default_factory=list)


class PlanAgentPipelineV2:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
        dependency_threshold: int = 2,
        cost_step_threshold: int = 40,
        cost_token_threshold: int = 100000,
        skill_persist_path: str | None = None,
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time_per_subagent = max_time_per_subagent
        self.selective_fallback_rule = selective_fallback_rule
        self.llm_selection_strategy = llm_selection_strategy
        self.dependency_threshold = dependency_threshold
        self.cost_step_threshold = cost_step_threshold
        self.cost_token_threshold = cost_token_threshold
        self.skill_persist_path = skill_persist_path

        self.plan_agent = PlanAgentV2(
            cheap_llm_cfg=cheap_llm_cfg,
            expensive_llm_cfg=expensive_llm_cfg,
            max_planning_steps=max_planning_steps,
            llm_selection_strategy=llm_selection_strategy,
        )
        self.sub_agent = SubAgentV2(
            cheap_llm_cfg=cheap_llm_cfg,
            expensive_llm_cfg=expensive_llm_cfg,
            max_steps=max_steps_per_subagent,
            max_retries_per_call=max_retries_per_call,
            max_time=max_time_per_subagent,
            selective_fallback_rule=selective_fallback_rule,
        )
        self.detector = CostDetector(
            step_threshold=cost_step_threshold,
            token_threshold=cost_token_threshold,
        )
        self.evolver = SkillEvolver(llm_cfg=expensive_llm_cfg)

        self._registry = SkillRegistry()
        self._output_dir: str | None = None
        if skill_persist_path:
            self._registry.load_from_file(skill_persist_path)

    @staticmethod
    def scan_workspace(
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        max_files: int = 300,
        max_files_per_dir: int = 50,
        timeout: float = 60.0,
    ) -> str:
        from src.agent.code_agent_plan_mode import CodeAgentPlanMode
        return CodeAgentPlanMode.scan_workspace(
            workspace=workspace,
            repo_path=repo_path,
            max_files=max_files,
            max_files_per_dir=max_files_per_dir,
            timeout=timeout,
        )

    def _select_llm_rule_based(self, related_indices: list[int]) -> Literal["cheap", "expensive"]:
        if len(related_indices) >= self.dependency_threshold:
            logger.info(f"[COAT V2 Rule] {len(related_indices)} deps >= {self.dependency_threshold}, using expensive")
            return "expensive"
        return "cheap"

    def _select_llm_llm_judged(self, llm_choice_from_tool: str | None) -> Literal["cheap", "expensive"]:
        if llm_choice_from_tool in ("expensive", "cheap"):
            return llm_choice_from_tool
        return "cheap"

    def _try_evolve_skill(self, anomaly: CostAnomaly) -> tuple[Skill | None, str]:
        logger.info(
            f"[COAT V2 Evolution] Cost anomaly detected for subtask #{anomaly.subtask_index}: "
            f"{anomaly.total_steps} steps (threshold={anomaly.threshold}). Evolving new skill..."
        )
        skill, skip_reason = self.evolver.evolve(anomaly)
        if skill:
            self._registry.register(skill)
            if self.skill_persist_path:
                self._registry.save_to_file(self.skill_persist_path)
            if self._output_dir:
                self._registry.export_skill_files(self._output_dir)
            logger.info(f"[COAT V2 Evolution] New skill registered: {skill.name}")
        else:
            logger.info(f"[COAT V2 Evolution] No skill generated: {skip_reason}")
        return skill, skip_reason

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PlanAgentResultV2:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_plan_v2")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)
        self._output_dir = out_dir

        logger.info(
            f"[COAT V2 Pipeline] Starting | strategy={self.llm_selection_strategy} | "
            f"existing_skills={self._registry.skill_names()} | "
            f"cost_threshold={self.cost_step_threshold} steps"
        )

        logger.info("[COAT V2 Pipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[COAT V2 Pipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResultV1] = []
        completed_subtasks: list[dict] = []
        evolved_skills: list[str] = []
        cost_anomalies: list[dict] = []

        planning_usage_before = self.plan_agent.caller.get_total_usage()

        messages: list[dict] = [
            {"role": "system", "content": self.plan_agent._build_system_prompt([])},
            {"role": "user", "content": task_description},
        ]

        if workspace_overview:
            messages.append({
                "role": "user",
                "content": f"## Workspace Overview\n{workspace_overview[:6000]}",
            })

        create_subagent_count = 0
        total_step_count = 0
        terminated = False

        while total_step_count < self.max_planning_total_steps and not terminated:
            total_step_count += 1
            logger.info(
                f"[COAT V2 Pipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
                f"CreateSubagent used: {create_subagent_count}/{self.max_planning_steps}"
            )

            response_msg = self.plan_agent._call_llm(messages)
            assistant_msg = self._response_to_dict(response_msg)
            messages.append(assistant_msg)

            tool_calls = response_msg.tool_calls or []
            if not tool_calls:
                remaining_total = self.max_planning_total_steps - total_step_count
                remaining_subagent = self.max_planning_steps - create_subagent_count
                messages.append({
                    "role": "user",
                    "content": (
                        f"Please use the available tools to proceed. "
                        f"({remaining_total} total steps remaining, "
                        f"{remaining_subagent} CreateSubagent calls remaining)"
                    ),
                })
                continue

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                if tool_name == "terminate":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Task terminated.",
                    })
                    logger.info("[COAT V2 Pipeline] Planning agent terminated.")
                    terminated = True
                    break

                elif tool_name == "CreateSubagent":
                    if create_subagent_count >= self.max_planning_steps:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"Error: Maximum number of CreateSubagent calls "
                                f"({self.max_planning_steps}) reached. "
                                f"Please call `terminate` to finish."
                            ),
                        })
                        continue

                    create_subagent_count += 1
                    subtask = tool_args.get("subtask", "")
                    related_indices = tool_args.get("related_subtask_index", [])
                    llm_choice_from_tool = tool_args.get("llm_choice", None)

                    if self.llm_selection_strategy == "rule_based":
                        llm_choice = self._select_llm_rule_based(related_indices)
                    else:
                        llm_choice = self._select_llm_llm_judged(llm_choice_from_tool)

                    subtask_index = len(subtask_results) + 1
                    logger.info(
                        f"[COAT V2 Pipeline] Creating subtask #{subtask_index}: "
                        f"{subtask[:100]} | related={related_indices} | llm={llm_choice}"
                    )

                    related_trajectories = self._build_related_trajectories(
                        related_indices, subtask_results,
                    )

                    sub_output_dir = os.path.join(logs_dir, f"subtask_{subtask_index}")
                    os.makedirs(sub_output_dir, exist_ok=True)

                    result = self.sub_agent.run(
                        original_task=instruction,
                        subtask_description=subtask,
                        related_trajectories=related_trajectories,
                        workspace=workspace,
                        repo_path=repo_path,
                        callbacks=callbacks,
                        output_dir=sub_output_dir,
                        llm_choice=llm_choice,
                    )
                    result.index = subtask_index
                    result.related_subtask_index = related_indices

                    subtask_results.append(result)

                    # --- Detection Module ---
                    anomaly = self.detector.check(
                        subtask_index=subtask_index,
                        subtask_description=subtask,
                        trajectory_records=result.trajectory_records,
                        metrics=result.metrics,
                    )
                    if anomaly:
                        anomaly_dict = {
                            "subtask_index": anomaly.subtask_index,
                            "subtask_description": anomaly.subtask_description,
                            "total_steps": anomaly.total_steps,
                            "threshold": anomaly.threshold,
                        }
                        cost_anomalies.append(anomaly_dict)
                        logger.info(
                            f"[COAT V2 Pipeline] Cost anomaly: subtask #{subtask_index} "
                            f"used {anomaly.total_steps} steps"
                        )

                        # --- Self-Evolution Module ---
                        new_skill, skip_reason = self._try_evolve_skill(anomaly)
                        if new_skill:
                            evolved_skills.append(new_skill.name)
                            anomaly_dict["skill_evolved"] = new_skill.name
                            self._registry.deploy_to_workspace(workspace, repo_path)
                        else:
                            anomaly_dict["skill_evolved"] = None
                            anomaly_dict["skip_reason"] = skip_reason

                    completed_subtasks.append({
                        "index": subtask_index,
                        "subtask": subtask,
                        "error": result.error,
                        "completed_normally": result.completed_normally,
                    })

                    new_system = self.plan_agent._build_system_prompt(completed_subtasks)
                    messages[0] = {"role": "system", "content": new_system}

                    obs_content = _serialize_subagent_trajectory(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs_content,
                    })

                    logger.info(
                        f"[COAT V2 Pipeline] Subtask #{subtask_index} done: "
                        f"completed_normally={result.completed_normally}, "
                        f"useful_indexes={result.useful_trajectory_indexes}, "
                        f"tokens={result.metrics.get('total_tokens', 0)}"
                    )

                elif tool_name == "view_file":
                    obs = _exec_view_file(tool_args, workspace)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs,
                    })

                elif tool_name == "search_by_keyword":
                    obs = _exec_search_by_keyword(tool_args, workspace)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs,
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tool_name}",
                    })

            if terminated:
                break

        if not terminated:
            logger.warning("[COAT V2 Pipeline] Reached max total planning steps without terminate")

        planning_usage_after = self.plan_agent.caller.get_total_usage()
        planning_metrics = {
            "prompt_tokens": planning_usage_after["input_tokens"] - planning_usage_before["input_tokens"],
            "completion_tokens": planning_usage_after["output_tokens"] - planning_usage_before["output_tokens"],
            "cache_read_tokens": planning_usage_after["cached_tokens"] - planning_usage_before["cached_tokens"],
            "reasoning_tokens": planning_usage_after["reasoning_tokens"] - planning_usage_before["reasoning_tokens"],
            "total_tokens": planning_usage_after["total_tokens"] - planning_usage_before["total_tokens"],
            "accumulated_cost": 0.0,
            "llm_used": "expensive",
        }

        exec_metrics_list = [r.metrics for r in subtask_results]
        exec_total = self._sum_metrics(exec_metrics_list)

        cheap_metrics_list = [r.metrics for r in subtask_results if r.llm_used == "cheap"]
        expensive_metrics_list = [r.metrics for r in subtask_results if r.llm_used == "expensive"]
        cheap_exec_total = self._sum_metrics(cheap_metrics_list)
        expensive_exec_total = self._sum_metrics(expensive_metrics_list)

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "execution_cheap": cheap_exec_total,
            "execution_expensive": expensive_exec_total,
            "total": self._merge_metrics(planning_metrics, exec_total),
            "total_cheap": cheap_exec_total,
            "total_expensive": self._merge_metrics(planning_metrics, expensive_exec_total),
            "evolution_summary": {
                "total_anomalies": len(cost_anomalies),
                "skills_evolved": evolved_skills,
                "total_skills_available": len(self._registry.skill_names()),
            },
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(subtask_results),
            "evolved_skills": evolved_skills,
            "cost_anomalies": cost_anomalies,
            "subtasks": [
                {
                    "index": r.index,
                    "subtask": r.subtask,
                    "related_subtask_index": r.related_subtask_index,
                    "error": r.error,
                    "completed_normally": r.completed_normally,
                    "useful_trajectory_indexes": r.useful_trajectory_indexes,
                    "files_modified": r.files_modified,
                    "llm_used": r.llm_used,
                    "metrics": r.metrics,
                }
                for r in subtask_results
            ],
            "metrics_summary": total_metrics,
        })

        if self.skill_persist_path:
            self._registry.save_to_file(self.skill_persist_path)

        self._registry.export_skill_files(out_dir)

        return PlanAgentResultV2(
            subtask_results=subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
            evolved_skills=evolved_skills,
            cost_anomalies=cost_anomalies,
        )

    def _build_related_trajectories(
        self,
        related_indices: list[int],
        subtask_results: list[SubtaskResultV1],
    ) -> list[dict]:
        result_map = {r.index: r for r in subtask_results}
        trajectories = []
        for idx in related_indices:
            prev_result = result_map.get(idx)
            if prev_result is None:
                logger.warning(f"[COAT V2 Pipeline] Related subtask #{idx} not found, skipping")
                continue
            filtered_text = _filter_trajectory_records(
                prev_result.trajectory_records,
                prev_result.useful_trajectory_indexes,
            )
            filtered_records = []
            if prev_result.useful_trajectory_indexes and prev_result.trajectory_records:
                useful_set = set(prev_result.useful_trajectory_indexes)
                for rec in prev_result.trajectory_records:
                    if rec.get("role") in ("tool", "assistant") and rec.get("index", -1) in useful_set:
                        filtered_records.append(rec)
            trajectories.append({
                "index": idx,
                "subtask": prev_result.subtask,
                "filtered_trajectory": filtered_text,
                "filtered_records": filtered_records,
            })
        return trajectories

    @staticmethod
    def _response_to_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant"}
        d["content"] = msg.content if msg.content else None
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content:
            d["reasoning_content"] = reasoning_content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    @staticmethod
    def _sum_metrics(metrics_list: list[dict]) -> dict[str, Any]:
        keys = ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                "reasoning_tokens", "total_tokens")
        total = {k: 0 for k in keys} | {"accumulated_cost": 0.0}
        for m in metrics_list:
            for k in keys:
                total[k] += m.get(k, 0)
            total["accumulated_cost"] += m.get("accumulated_cost", 0.0)
        return total

    @staticmethod
    def _merge_metrics(a: dict, b: dict) -> dict[str, Any]:
        keys = ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                "reasoning_tokens", "total_tokens")
        result = {k: 0 for k in keys} | {"accumulated_cost": 0.0}
        for k in keys:
            result[k] += a.get(k, 0) + b.get(k, 0)
        result["accumulated_cost"] += a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0)
        return result

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
