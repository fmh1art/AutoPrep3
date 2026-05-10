"""
COAT V4 — Live Self-Evolving Planning Agent 框架。

COAT: Context-aware Orchestrated Agent with Tool-calling

V4 相比 V3 的核心改进 — 借鉴 Live-SWE-agent 的自进化范式：

  V3 的问题：
    - 进化时机被动：只在 CostDetector 检测到异常（steps>=40）后才触发
    - 进化主体分离：SkillEvolver 是独立的 LLM 调用，与 Execution Agent 脱节
    - Skill 质量不可控：Evolver 只看到轨迹摘要，缺乏对当前任务的深度理解
    - Skill 使用率低：LLM 倾向于直接用 bash 命令而非调用 skill

  V4 的改进：
    1. 运行时反思触发（Runtime Reflection）：
       - Execution Agent 每收到 observation 后，自动反思是否需要创建工具
       - 不再依赖 CostDetector 被动触发
    2. Agent 自主工具创建（On-the-Fly Tool Creation）：
       - Execution Agent 自己决定何时创建工具、创建什么工具
       - 工具是 Python 脚本，Agent 直接写入 /workspace/.skill/ 目录
       - 无需独立的 SkillEvolver LLM 调用
    3. 工具积累与共享（Skill Accumulation）：
       - 保留 SkillRegistryV4，跨 subtask 积累工具
       - 新 subtask 的 system prompt 包含所有已有工具的描述
       - 工具文件持久化到 workspace，无需重复部署
    4. 保留 V3 的基本流程：
       - Planning Agent + Execution Agent 两层架构
       - Selective Context (useful_trajectory_indexes)
       - Planning Agent 不感知 skill（skill 是执行层面的事）

与 V3 的代码差异：
    - 移除 CostDetectorV3、SkillEvolverV3
    - SelfEvolveCodeAgentV4 的 observation 后附加反思提示
    - SelfEvolveCodeAgentV4 的 system prompt 包含工具创建指南
    - SkillRegistryV4 简化：只做注册、部署、prompt 构建
"""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent_optimized import AgentResult
from src.agent.plan_agent import (
    PLAN_AGENT_TOOLS,
    SubtaskResult,
    PlanAgentResult,
    _filter_trajectory_records,
    _serialize_subagent_trajectory,
    _exec_view_file,
    _exec_search_by_keyword,
)
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COAT V4 Skill 定义与注册表
# ---------------------------------------------------------------------------

@dataclass
class SkillV4:
    name: str
    description: str
    script_path: str
    script_content: str = ""
    bash_script: str = ""
    use_count: int = 0
    created_at: float = 0.0
    source_subtask: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "script_path": self.script_path,
            "script_content": self.script_content,
            "bash_script": self.bash_script,
            "use_count": self.use_count,
            "created_at": self.created_at,
            "source_subtask": self.source_subtask,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillV4":
        return cls(
            name=d["name"],
            description=d["description"],
            script_path=d["script_path"],
            script_content=d.get("script_content", ""),
            bash_script=d.get("bash_script", ""),
            use_count=d.get("use_count", 0),
            created_at=d.get("created_at", 0.0),
            source_subtask=d.get("source_subtask", ""),
        )


class SkillRegistryV4:
    _instance: "SkillRegistryV4 | None" = None
    MAX_SKILLS = 10

    def __new__(cls) -> "SkillRegistryV4":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: dict[str, SkillV4] = {}
        return cls._instance

    def register(self, skill: SkillV4) -> None:
        if len(self._skills) >= self.MAX_SKILLS and skill.name not in self._skills:
            self.evict_least_used()
        self._skills[skill.name] = skill
        logger.info(f"[SkillRegistryV4] Registered skill: {skill.name} (total: {len(self._skills)})")

    def increment_use_count(self, skill_name: str) -> None:
        skill = self._skills.get(skill_name)
        if skill is not None:
            skill.use_count += 1

    def evict_least_used(self) -> str | None:
        if not self._skills:
            return None
        least_name = min(self._skills, key=lambda n: self._skills[n].use_count)
        least_skill = self._skills.pop(least_name)
        logger.info(
            f"[SkillRegistryV4] Evicted skill '{least_name}' "
            f"(use_count={least_skill.use_count}) to make room"
        )
        return least_name

    def get(self, name: str) -> SkillV4 | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillV4]:
        return list(self._skills.values())

    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    SKILL_DIR = "/workspace/.skill"

    def scan_workspace_skills(self, workspace: "DockerWorkspace", repo_path: str = "/workspace") -> set[str]:
        skill_dir = self.SKILL_DIR
        changed_skills: set[str] = set()

        ls_result = workspace.execute_command(
            f"ls {skill_dir}/ 2>/dev/null", timeout=10
        )
        if ls_result.exit_code != 0 or not ls_result.stdout.strip():
            return changed_skills

        files_in_dir = [f.strip() for f in ls_result.stdout.strip().split("\n") if f.strip()]
        py_files = {f[:-3] for f in files_in_dir if f.endswith(".py")}
        sh_files = {f[:-3] for f in files_in_dir if f.endswith(".sh")}
        all_names = py_files | sh_files

        for name in all_names:
            description = ""
            script_content = ""
            bash_script = ""

            if name in py_files:
                py_path = f"{skill_dir}/{name}.py"
                desc_result = workspace.execute_command(
                    f"head -5 {py_path} 2>/dev/null", timeout=10
                )
                if desc_result.exit_code == 0 and desc_result.stdout.strip():
                    for dline in desc_result.stdout.strip().split("\n"):
                        dline = dline.strip()
                        if dline.startswith('#') and not dline.startswith('#!') and not dline.startswith('# -*-'):
                            description = dline.lstrip('#').strip()
                            break
                content_result = workspace.execute_command(
                    f"cat {py_path} 2>/dev/null", timeout=10
                )
                script_content = content_result.stdout if content_result.exit_code == 0 else ""

            if name in sh_files:
                sh_path = f"{skill_dir}/{name}.sh"
                if not description:
                    desc_result = workspace.execute_command(
                        f"head -5 {sh_path} 2>/dev/null", timeout=10
                    )
                    if desc_result.exit_code == 0 and desc_result.stdout.strip():
                        for dline in desc_result.stdout.strip().split("\n"):
                            dline = dline.strip()
                            if dline.startswith('#') and not dline.startswith('#!'):
                                description = dline.lstrip('#').strip()
                                break
                content_result = workspace.execute_command(
                    f"cat {sh_path} 2>/dev/null", timeout=10
                )
                bash_script = content_result.stdout if content_result.exit_code == 0 else ""

            if not description:
                description = f"Custom tool: {name}"

            existing = self._skills.get(name)
            if existing is not None:
                if (existing.script_content != script_content or existing.bash_script != bash_script):
                    existing.script_content = script_content
                    existing.bash_script = bash_script
                    existing.description = description
                    changed_skills.add(name)
                    logger.info(f"[SkillRegistryV4] Skill updated: {name}")
            else:
                self.register(SkillV4(
                    name=name,
                    description=description,
                    script_path=f"{skill_dir}/{name}",
                    script_content=script_content,
                    bash_script=bash_script,
                    source_subtask="auto_scanned",
                ))
                changed_skills.add(name)

        return changed_skills

    def deploy_to_workspace(self, workspace: "DockerWorkspace", repo_path: str = "/workspace") -> None:
        skill_dir = self.SKILL_DIR
        workspace.execute_command(f"mkdir -p {skill_dir}", timeout=10)
        for skill in self._skills.values():
            if skill.script_content:
                py_path = f"{skill_dir}/{skill.name}.py"
                encoded = base64.b64encode(skill.script_content.encode("utf-8")).decode("ascii")
                workspace.execute_command(
                    f"echo {encoded} | base64 -d > {py_path}", timeout=10,
                )
                workspace.execute_command(f"chmod +x {py_path}", timeout=10)
            if skill.bash_script:
                sh_path = f"{skill_dir}/{skill.name}.sh"
                encoded = base64.b64encode(skill.bash_script.encode("utf-8")).decode("ascii")
                workspace.execute_command(
                    f"echo {encoded} | base64 -d > {sh_path}", timeout=10,
                )
                workspace.execute_command(f"chmod +x {sh_path}", timeout=10)
        active_names = set(self._skills.keys())
        ls_result = workspace.execute_command(f"ls {skill_dir}/ 2>/dev/null", timeout=10)
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for fname in ls_result.stdout.strip().split("\n"):
                fname = fname.strip()
                for ext in (".py", ".sh"):
                    if fname.endswith(ext):
                        sname = fname[:-3]
                        if sname not in active_names:
                            workspace.execute_command(
                                f"rm -f {skill_dir}/{fname}", timeout=10,
                            )
                            logger.info(f"[SkillRegistryV4] Cleaned up evicted file: {fname}")

    def build_skill_prompt_section(self) -> str:
        if not self._skills:
            return ""
        parts = [
            "## Available Skills\n",
            "You have the following skills available. Each skill may have a bash wrapper (`.sh`) "
            "and/or a Python script (`.py`) in `/workspace/.skill/`.\n",
            "Call them via bash:\n",
            "- `bash /workspace/.skill/<name>.sh <args>` (if .sh exists)\n",
            "- `python3 /workspace/.skill/<name>.py <args>` (if .py exists)\n\n",
            "**IMPORTANT: Prefer using available skills over manual multi-step bash commands.** "
            "Skills encapsulate common multi-step operations and can save significant steps. "
            "Before writing a sequence of bash commands, check if an available skill already does what you need.\n",
        ]
        for skill in self._skills.values():
            parts.append(f"### {skill.name}\n{skill.description}\n")
        return "\n".join(parts)

    def save_to_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        existing_data: dict = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_data = {}
        for name, skill in self._skills.items():
            existing_data[name] = skill.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def export_skill_files(self, output_dir: str) -> None:
        skills_dir = os.path.join(output_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        manifest_path = os.path.join(skills_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({n: s.to_dict() for n, s in self._skills.items()}, f, ensure_ascii=False, indent=2)
        if self._skills:
            logger.info(f"[SkillRegistryV4] Skill manifest exported to {manifest_path}")

    def load_from_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, d in data.items():
            self._skills[name] = SkillV4.from_dict(d)
        logger.info(f"[SkillRegistryV4] Loaded {len(data)} skills from {path}")

    def reset(self) -> None:
        self._skills.clear()


# ---------------------------------------------------------------------------
# COAT V4 Execution Agent — bash + finish + runtime reflection
# ---------------------------------------------------------------------------

BASH_ONLY_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in the Docker workspace. Use `&&` or `;` "
            "to chain multiple commands. You can also call skill scripts "
            "via `bash /workspace/.skill/<name>.sh <args>` or "
            "`python3 /workspace/.skill/<name>.py <args>`"
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


def _build_execution_system_prompt_v4(repo_path: str, skill_prompt: str) -> str:
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
        "",
        "## Creating and Modifying Tools",
        "",
        "You can create new tools or modify existing ones to help with your workflow. "
        "Compared to basic bash commands, custom tools can better aid your workflow by encapsulating multi-step operations.",
        "",
        "### When to Create or Modify a Tool",
        "- You find yourself repeating a multi-step operation (e.g., search + filter + format)",
        "- You need to perform a complex edit that requires careful line management",
        "- You need to parse or transform data in a specific way repeatedly",
        "- Basic bash commands are insufficient or error-prone for the task",
        "- An existing tool almost does what you need but requires adjustment",
        "",
        "### How to Create a Tool",
        "Each tool consists of a bash wrapper script (`.sh`) and optionally a Python script (`.py`):",
        "```bash",
        "# Create the bash wrapper (required)",
        "cat <<'SHEOF' > /workspace/.skill/<tool_name>.sh",
        "#!/bin/bash",
        "# Description: <what this tool does, its parameters, and usage>",
        "python3 \"$(dirname \"$0\")/<tool_name>.py\" \"$@\"",
        "SHEOF",
        "chmod +x /workspace/.skill/<tool_name>.sh",
        "",
        "# Create the Python script (required for most tools)",
        "cat <<'PYEOF' > /workspace/.skill/<tool_name>.py",
        "# Description: <what this tool does, its parameters, and usage>",
        "#!/usr/bin/env python3",
        "import sys",
        "import os",
        "# ... your implementation ...",
        "if __name__ == '__main__':",
        "    main()",
        "PYEOF",
        "chmod +x /workspace/.skill/<tool_name>.py",
        "```",
        "",
        "### How to Modify an Existing Tool",
        "If an existing tool needs adjustment, simply overwrite its files:",
        "```bash",
        "# Edit the Python script",
        "cat <<'PYEOF' > /workspace/.skill/<tool_name>.py",
        "# Description: <updated description>",
        "#!/usr/bin/env python3",
        "# ... updated implementation ...",
        "PYEOF",
        "```",
        "",
        "### Tool Requirements",
        "- The first comment line must be a description starting with `# Description:`",
        "- Use only standard library + common packages (os, sys, json, re, subprocess, argparse)",
        "- Include informative outputs or error messages",
        "- Make the tool general enough to be reusable, not tied to this specific task",
        "- After creating or modifying a tool, you can use it: `bash /workspace/.skill/<name>.sh <args>` or `python3 /workspace/.skill/<name>.py <args>`",
    ])
    return "\n".join(parts)


REFLECTION_SUFFIX = (
    "\n\n---\n"
    "Reflect on the previous trajectories and decide if there are any tools you can create "
    "or modify to help you with the current task. "
    "If you notice repetitive multi-step operations or complex tasks that would benefit from a dedicated tool, "
    "create one by writing scripts to `/workspace/.skill/<name>.sh` and `/workspace/.skill/<name>.py`. "
    "If an existing tool almost does what you need, modify it by overwriting its files. "
    "Note that just because you can use basic bash commands doesn't mean you should not create tools that can still be helpful."
)


class SelfEvolveCodeAgentV4:
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
        registry = SkillRegistryV4()
        skill_prompt = registry.build_skill_prompt_section()
        return _build_execution_system_prompt_v4(repo_path, skill_prompt)

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
                logger.warning(f"[SelfEvolveCodeAgentV4] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"SelfEvolveCodeAgentV4 LLM call failed after 3 attempts") from last_exc

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_sev_v4")
        os.makedirs(out_dir, exist_ok=True)

        registry = SkillRegistryV4()
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
                logger.warning(f"[SelfEvolveCodeAgentV4] Timeout at step {step}")
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

                    self._detect_new_skills(command, workspace, repo_path)
                else:
                    observation = f"Error: Unknown tool '{tool_name}'"

                trajectory.append({
                    "index": step,
                    "role": "tool",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "observation": observation,
                })

                tool_msg_content = f"{observation}\n[Step {step}]"
                if not finished and tool_name == "bash":
                    tool_msg_content += REFLECTION_SUFFIX

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_msg_content,
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
            logger.warning("[SelfEvolveCodeAgentV4] Reached max steps without finishing")

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

    def _detect_new_skills(self, command: str, workspace: "DockerWorkspace", repo_path: str = "/workspace") -> None:
        if ".skill/" not in command:
            return
        registry = SkillRegistryV4()
        for skill_name in registry.skill_names():
            if f".skill/{skill_name}" in command:
                registry.increment_use_count(skill_name)
        names_before = set(registry.skill_names())
        changed = registry.scan_workspace_skills(workspace, repo_path)
        if changed:
            logger.info(f"[SelfEvolveCodeAgentV4] Skills changed: {changed}")
        names_after = set(registry.skill_names())
        evicted = names_before - names_after
        if evicted:
            skill_dir = SkillRegistryV4.SKILL_DIR
            for evicted_name in evicted:
                workspace.execute_command(
                    f"rm -f {skill_dir}/{evicted_name}.py {skill_dir}/{evicted_name}.sh 2>/dev/null",
                    timeout=10,
                )
                logger.info(f"[SelfEvolveCodeAgentV4] Cleaned up evicted skill files: {evicted_name}")

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
        lines.append(f"# Execution Agent (SelfEvolve V4) LLM IO — Step {step}\n")
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
# COAT V4 SubAgentV4 — wraps SelfEvolveCodeAgentV4
# ---------------------------------------------------------------------------

class SubAgentV4:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
    ):
        self.llm_cfg = llm_cfg
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
    ) -> SubtaskResult:
        agent = SelfEvolveCodeAgentV4(
            llm_cfg=self.llm_cfg,
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
            f"[COAT V4 SubAgent] Running subtask: {subtask_description[:100]} | "
            f"skills={SkillRegistryV4().skill_names()}"
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

            return SubtaskResult(
                subtask=subtask_description,
                useful_trajectory_indexes=useful_indexes,
                messages=result.messages if hasattr(result, "messages") else [],
                trajectory_records=trajectory_records,
                metrics=result.metrics if hasattr(result, "metrics") else {},
                completed_normally=completed_normally and not hit_max_steps,
                hit_max_steps=hit_max_steps,
            )
        except Exception as e:
            logger.error(f"[COAT V4 SubAgent] Execution failed: {e}")
            return SubtaskResult(
                subtask=subtask_description,
                error=str(e),
                completed_normally=False,
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
# COAT V4 PlanAgentV4 — planning agent (same as V0/V3, single LLM)
# ---------------------------------------------------------------------------

class PlanAgentV4:
    def __init__(self, llm_cfg: dict, max_planning_steps: int = 20):
        self.llm_cfg = llm_cfg
        self.max_planning_steps = max_planning_steps
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    @property
    def caller(self):
        return self._caller

    def _build_system_prompt(self, completed_subtasks: list[dict]) -> str:
        return render_j2("plan_agent_step_by_step.j2", context={
            "completed_subtasks": completed_subtasks,
        })

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(
                    messages=messages,
                    tools=PLAN_AGENT_TOOLS,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[COAT V4 PlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"COAT V4 PlanAgent LLM call failed after 3 attempts") from last_exc


# ---------------------------------------------------------------------------
# COAT V4 PlanAgentPipelineV4 — full pipeline with live self-evolution
# ---------------------------------------------------------------------------

@dataclass
class PlanAgentResultV4:
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)
    evolved_skills: list[str] = field(default_factory=list)


class PlanAgentPipelineV4:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
        skill_persist_path: str | None = None,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time_per_subagent = max_time_per_subagent
        self.selective_fallback_rule = selective_fallback_rule
        self.skill_persist_path = skill_persist_path

        self.plan_agent = PlanAgentV4(
            llm_cfg=llm_cfg,
            max_planning_steps=max_planning_steps,
        )
        self.sub_agent = SubAgentV4(
            llm_cfg=llm_cfg,
            max_steps=max_steps_per_subagent,
            max_retries_per_call=max_retries_per_call,
            max_time=max_time_per_subagent,
            selective_fallback_rule=selective_fallback_rule,
        )

        self._registry = SkillRegistryV4()
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

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PlanAgentResultV4:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_plan_v4")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)
        self._output_dir = out_dir

        self._registry.reset()
        if self.skill_persist_path and os.path.isfile(self.skill_persist_path):
            self._registry.load_from_file(self.skill_persist_path)
            logger.info(
                f"[COAT V4 Pipeline] Loaded {len(self._registry.skill_names())} "
                f"skills from {self.skill_persist_path}"
            )

        self._registry.deploy_to_workspace(workspace, repo_path)
        self._registry.scan_workspace_skills(workspace, repo_path)

        logger.info(
            f"[COAT V4 Pipeline] Starting | "
            f"initial_skills={self._registry.skill_names()}"
        )

        logger.info("[COAT V4 Pipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[COAT V4 Pipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResult] = []
        completed_subtasks: list[dict] = []
        evolved_skills: list[str] = []

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
                f"[COAT V4 Pipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
                f"CreateSubagent used: {create_subagent_count}/{self.max_planning_steps}"
            )

            response_msg = self.plan_agent._call_llm(messages)
            assistant_msg = self._response_to_dict(response_msg)
            messages.append(assistant_msg)

            self._save_llm_io(logs_dir, total_step_count - 1, messages, response_msg)

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
                    logger.info("[COAT V4 Pipeline] Planning agent terminated.")
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

                    subtask_index = len(subtask_results) + 1
                    logger.info(
                        f"[COAT V4 Pipeline] Creating subtask #{subtask_index}: "
                        f"{subtask[:100]} | related={related_indices}"
                    )

                    related_trajectories = self._build_related_trajectories(
                        related_indices, subtask_results,
                    )

                    sub_output_dir = os.path.join(logs_dir, f"subtask_{subtask_index}")
                    os.makedirs(sub_output_dir, exist_ok=True)

                    skills_before = set(self._registry.skill_names())

                    result = self.sub_agent.run(
                        original_task=instruction,
                        subtask_description=subtask,
                        related_trajectories=related_trajectories,
                        workspace=workspace,
                        repo_path=repo_path,
                        callbacks=callbacks,
                        output_dir=sub_output_dir,
                    )
                    result.index = subtask_index
                    result.related_subtask_index = related_indices

                    subtask_results.append(result)

                    skills_after = set(self._registry.skill_names())
                    new_skills = skills_after - skills_before
                    if new_skills:
                        evolved_skills.extend(sorted(new_skills))
                        logger.info(
                            f"[COAT V4 Pipeline] New skills evolved in subtask #{subtask_index}: {new_skills}"
                        )

                    completed_subtasks.append({
                        "index": subtask_index,
                        "subtask": subtask,
                        "error": result.error,
                        "completed_normally": result.completed_normally,
                    })

                    self._save_subtask_result(logs_dir, result)

                    new_system = self.plan_agent._build_system_prompt(completed_subtasks)
                    messages[0] = {"role": "system", "content": new_system}

                    obs_content = _serialize_subagent_trajectory(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs_content,
                    })

                    logger.info(
                        f"[COAT V4 Pipeline] Subtask #{subtask_index} done: "
                        f"completed_normally={result.completed_normally}, "
                        f"useful_indexes={result.useful_trajectory_indexes}, "
                        f"tokens={result.metrics.get('total_tokens', 0)}, "
                        f"new_skills={new_skills}"
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
            logger.warning("[COAT V4 Pipeline] Reached max total planning steps without terminate")

        planning_usage_after = self.plan_agent.caller.get_total_usage()
        planning_metrics = self._compute_usage_delta(planning_usage_before, planning_usage_after)

        exec_metrics_list = [r.metrics for r in subtask_results]
        exec_total = self._sum_metrics(exec_metrics_list)

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "total": self._merge_metrics(planning_metrics, exec_total),
            "evolution_summary": {
                "skills_evolved": evolved_skills,
                "total_skills_available": len(self._registry.skill_names()),
            },
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(subtask_results),
            "evolved_skills": evolved_skills,
            "subtasks": [
                {
                    "index": r.index,
                    "subtask": r.subtask,
                    "related_subtask_index": r.related_subtask_index,
                    "error": r.error,
                    "completed_normally": r.completed_normally,
                    "useful_trajectory_indexes": r.useful_trajectory_indexes,
                    "files_modified": r.files_modified,
                    "metrics": r.metrics,
                }
                for r in subtask_results
            ],
            "metrics_summary": total_metrics,
        })

        if self.skill_persist_path:
            self._registry.save_to_file(self.skill_persist_path)

        self._registry.export_skill_files(out_dir)

        return PlanAgentResultV4(
            subtask_results=subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
            evolved_skills=evolved_skills,
        )

    def _build_related_trajectories(
        self,
        related_indices: list[int],
        subtask_results: list[SubtaskResult],
    ) -> list[dict]:
        result_map = {r.index: r for r in subtask_results}
        trajectories = []
        for idx in related_indices:
            prev_result = result_map.get(idx)
            if prev_result is None:
                logger.warning(f"[COAT V4 Pipeline] Related subtask #{idx} not found, skipping")
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

    def _save_llm_io(self, logs_dir: str, step: int, messages: list[dict], response_msg) -> None:
        llm_log_dir = os.path.join(logs_dir, "llm_io")
        os.makedirs(llm_log_dir, exist_ok=True)
        md_path = os.path.join(llm_log_dir, f"step_{step:03d}.md")

        lines: list[str] = []
        lines.append(f"# COAT V4 Planning Agent LLM IO — Step {step}\n")
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
                        args_display = {k: v for k, v in args_parsed.items() if k != "_trajectory_step_index"}
                        lines.append("```json\n" + json.dumps(args_display, ensure_ascii=False, indent=2) + "\n```\n")
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

    def _save_subtask_result(self, logs_dir: str, result: SubtaskResult):
        sub_dir = os.path.join(logs_dir, f"subtask_{result.index}")
        os.makedirs(sub_dir, exist_ok=True)

        result_data = {
            "index": result.index,
            "subtask": result.subtask,
            "related_subtask_index": result.related_subtask_index,
            "useful_trajectory_indexes": result.useful_trajectory_indexes,
            "error": result.error,
            "completed_normally": result.completed_normally,
            "files_modified": result.files_modified,
            "metrics": result.metrics,
        }
        self._write_json(os.path.join(sub_dir, "result.json"), result_data)

        if result.messages:
            try:
                with open(os.path.join(sub_dir, "messages.json"), "w", encoding="utf-8") as f:
                    json.dump(result.messages, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save messages for subtask {result.index}: {e}")

        if result.trajectory_records:
            try:
                with open(os.path.join(sub_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                    json.dump(result.trajectory_records, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save trajectory for subtask {result.index}: {e}")

    @staticmethod
    def _compute_usage_delta(before: dict, after: dict) -> dict[str, Any]:
        return {
            "prompt_tokens": after["input_tokens"] - before["input_tokens"],
            "completion_tokens": after["output_tokens"] - before["output_tokens"],
            "cache_read_tokens": after["cached_tokens"] - before["cached_tokens"],
            "reasoning_tokens": after["reasoning_tokens"] - before["reasoning_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "accumulated_cost": 0.0,
        }

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
