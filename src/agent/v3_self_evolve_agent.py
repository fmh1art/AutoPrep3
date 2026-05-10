from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent_optimized import AgentResult
from src.agent.v3_runtime_detector import RuntimeDetectorV3
from src.agent.v3_skill_hub import SkillHubManager
from src.agent.v3_skills import SkillRegistryV3
from src.agent.v3_utils import response_to_dict, save_llm_io
from src.module.gpt_inference import SimpleAPICaller

logger = logging.getLogger(__name__)


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


def _build_self_evolve_system_prompt_v3(repo_path: str, skill_prompt: str) -> str:
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


class SelfEvolveCodeAgentV3:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        skill_hub: SkillHubManager | None = None,
        runtime_detector: RuntimeDetectorV3 | None = None,
        bash_observation_threshold: int = 15000,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self._skill_hub = skill_hub
        self._runtime_detector = runtime_detector
        self.bash_observation_threshold = bash_observation_threshold
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )
        self._tools = [BASH_ONLY_TOOL, FINISH_TOOL]

    def _build_system_prompt(self, repo_path: str, subtask_description: str = "") -> str:
        if self._skill_hub and subtask_description:
            skill_prompt = self._skill_hub.build_skill_prompt_for_subtask(subtask_description)
        else:
            registry = SkillRegistryV3()
            skill_prompt = registry.build_skill_prompt_section()
        return _build_self_evolve_system_prompt_v3(repo_path, skill_prompt)

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
                logger.warning(f"[SelfEvolveCodeAgentV3] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"SelfEvolveCodeAgentV3 LLM call failed after 3 attempts") from last_exc

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        subtask_description: str = "",
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_sev_v3")
        os.makedirs(out_dir, exist_ok=True)

        registry = SkillRegistryV3()
        registry.deploy_to_workspace(workspace, repo_path)

        system_prompt = self._build_system_prompt(repo_path, subtask_description)
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
                logger.warning(f"[SelfEvolveCodeAgentV3] Timeout at step {step}")
                break

            response_msg = self._call_llm(messages)
            assistant_msg = response_to_dict(response_msg)
            messages.append(assistant_msg)

            save_llm_io(out_dir, step, messages, response_msg,
                        title_prefix="Execution Agent (SelfEvolve V3) LLM IO")

            tool_calls = response_msg.tool_calls or []
            if not tool_calls:
                remaining = self.max_steps - step - 1
                messages.append({
                    "role": "user",
                    "content": f"Please continue using the available tools. ({remaining} steps remaining)",
                })
                continue

            finished = False
            pending_runtime_messages: list[dict] = []
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                raw_observation: str | None = None

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
                                raw_observation = output + f"\n[exit code: {r.exit_code}]"
                                if len(output) > self.bash_observation_threshold:
                                    half = self.bash_observation_threshold // 2
                                    output = output[:half] + f"\n\n... ({len(output) - self.bash_observation_threshold} chars truncated) ...\n\n" + output[-half:]
                                observation = output + f"\n[exit code: {r.exit_code}]"
                        except Exception as e:
                            observation = f"Error: {e}"
                else:
                    observation = f"Error: Unknown tool '{tool_name}'"

                if self._runtime_detector is not None and not finished:
                    intervention = self._runtime_detector.check_long_observation(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=raw_observation if raw_observation is not None else observation,
                    )
                    if intervention is not None and intervention.revoke_step:
                        logger.info(
                            f"[SelfEvolveCodeAgentV3][RuntimeDetector] step={step} "
                            f"rule={intervention.rule} revoked (raw_obs_len="
                            f"{len(raw_observation) if raw_observation is not None else len(observation)})"
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"[Step {step}] Observation revoked by runtime advisor "
                                f"(length exceeded budget threshold)."
                            ),
                        })
                        pending_runtime_messages.append({
                            "role": "user",
                            "content": intervention.message,
                        })
                        continue

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

            if self._runtime_detector is not None and not finished:
                dup_intervention = self._runtime_detector.check_duplicate_tool_call(trajectory)
                if dup_intervention is not None:
                    logger.info(
                        f"[SelfEvolveCodeAgentV3][RuntimeDetector] step={step} "
                        f"rule={dup_intervention.rule} triggered"
                    )
                    pending_runtime_messages.append({
                        "role": "user",
                        "content": dup_intervention.message,
                    })

            for advisory in pending_runtime_messages:
                messages.append(advisory)

            if finished:
                break

            remaining = self.max_steps - step - 1
            if 0 < remaining <= 5:
                messages.append({
                    "role": "user",
                    "content": f"[WARNING] You have only {remaining} step(s) remaining. Call `finish` if done.",
                })
        else:
            logger.warning("[SelfEvolveCodeAgentV3] Reached max steps without finishing")

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
