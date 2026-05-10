from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.v3_skills import SkillV3
from src.agent.v3_utils import response_to_dict
from src.module.gpt_inference import SimpleAPICaller

logger = logging.getLogger(__name__)


SKILL_VALIDATION_SYSTEM_PROMPT = """You are a minimal code agent whose ONLY job is to verify that a skill script works correctly. You have access to only the `bash` tool and the `finish` tool.

You will be given a skill to validate. Your task:
1. Read the skill scripts to understand what the skill does.
2. Create minimal test data in /tmp/ if needed.
3. Run the skill with simple test arguments to verify it executes without errors.
4. Check that the skill produces reasonable output.
5. Clean up /tmp/ test files.
6. Call `finish` with validation_result='PASS' or 'FAIL'.

IMPORTANT RULES:
- Do NOT modify any files under /workspace/ except creating /workspace/.skill_test/ for testing.
- Do NOT install any packages or change the environment permanently.
- If the skill requires specific test files, create them in /tmp/ and clean up afterwards.
- If the skill's bash_script contains `pip install`, verify it works but note if it modifies the environment.
- Call `finish` when validation is complete, whether the skill passed or failed.
- Do NOT remove /workspace/.skill_test/ — that will be cleaned up externally.
"""

SKILL_VALIDATION_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the Docker workspace for skill validation.",
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

SKILL_VALIDATION_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call when skill validation is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "validation_result": {
                    "type": "string",
                    "enum": ["PASS", "FAIL"],
                    "description": "Whether the skill passed or failed validation.",
                },
                "validation_details": {
                    "type": "string",
                    "description": "Brief description of what was tested and the result.",
                },
            },
            "required": ["validation_result", "validation_details"],
        },
    },
}

SKILL_FIX_PROMPT = """You are a skill fixer. A skill failed validation. Based on the error details, fix the skill scripts.

## Skill: {skill_name}

### Description
{skill_description}

### Current Bash Script
```bash
{bash_script}
```

### Current Python Script
```python
{py_script}
```

### Validation Error
{validation_error}

## Fix Guidelines

1. Fix the specific error reported during validation.
2. Ensure the skill still meets ALL requirements:
   - Reusable across projects (no hardcoded task-specific values)
   - Fully parameterized (all paths/patterns via CLI args)
   - Multi-step workflow (3+ steps)
   - Self-contained (works in fresh Docker with stdlib)
   - Clear contract (usage syntax, parameters, output, example)
   - Dependency self-contained (pip install in bash_script if needed)
3. Handle edge cases gracefully (missing args, empty input, etc.).
4. Make sure the bash script properly passes all arguments to the Python script.
5. Make sure the Python script uses argparse or sys.argv correctly.

## Output Format

Respond with JSON only:
```json
{{
  "bash_script": "#!/bin/bash\\nset -e\\n...",
  "py_script": "#!/usr/bin/env python3\\nimport sys, argparse\\n...",
  "fix_description": "Brief description of what was fixed"
}}
```

If the skill is fundamentally broken and cannot be fixed, respond with:
```json
{{"skip": true, "reason": "explanation"}}
```"""


@dataclass
class SkillValidationResult:
    passed: bool
    details: str
    workspace_affected: bool = False
    fixed_skill: SkillV3 | None = None


class SkillValidator:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 15,
        max_fix_attempts: int = 2,
        max_retries_per_call: int = 3,
    ):
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )
        self.max_steps = max_steps
        self.max_fix_attempts = max_fix_attempts
        self.max_retries_per_call = max_retries_per_call
        self._tools = [SKILL_VALIDATION_BASH_TOOL, SKILL_VALIDATION_FINISH_TOOL]

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
                logger.warning(f"[SkillValidator] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"SkillValidator LLM call failed after 3 attempts") from last_exc

    def _snapshot_workspace(self, workspace: "DockerWorkspace") -> set[str]:
        r = workspace.execute_command(
            "find /workspace -maxdepth 3 -type f 2>/dev/null | sort",
            timeout=30,
        )
        return set((r.stdout or "").strip().splitlines())

    def _deploy_skill_to_test_dir(
        self,
        skill: SkillV3,
        workspace: "DockerWorkspace",
    ) -> tuple[bool, str]:
        import base64
        b64_bash = base64.b64encode(skill.bash_script.encode("utf-8")).decode("ascii")
        b64_py = base64.b64encode(skill.py_script.encode("utf-8")).decode("ascii")

        deploy_cmd = (
            f"mkdir -p /workspace/.skill_test && "
            f"echo '{b64_bash}' | base64 -d > /workspace/.skill_test/{skill.name}.sh && "
            f"chmod +x /workspace/.skill_test/{skill.name}.sh && "
            f"echo '{b64_py}' | base64 -d > /workspace/.skill_test/{skill.name}.py && "
            f"echo 'DEPLOY_OK'"
        )
        r = workspace.execute_command(deploy_cmd, timeout=10)
        if "DEPLOY_OK" not in (r.stdout or ""):
            return False, f"Failed to deploy: {r.stderr or r.stdout}"
        return True, ""

    def _run_validation_agent(
        self,
        skill: SkillV3,
        workspace: "DockerWorkspace",
    ) -> SkillValidationResult:
        user_prompt = (
            f"Please validate the following skill:\n\n"
            f"## Skill: {skill.name}\n"
            f"**Description:**\n{skill.description}\n\n"
            f"**Bash script** is at `/workspace/.skill_test/{skill.name}.sh`\n"
            f"**Python script** is at `/workspace/.skill_test/{skill.name}.py`\n\n"
            f"Steps:\n"
            f"1. Read the scripts to understand what the skill does.\n"
            f"2. Create minimal test data in /tmp/ if needed.\n"
            f"3. Run the skill with simple test arguments using: `bash /workspace/.skill_test/{skill.name}.sh <args>`\n"
            f"4. Verify the output is reasonable and there are no errors.\n"
            f"5. Clean up /tmp/ test files.\n"
            f"6. Call `finish` with validation_result='PASS' or 'FAIL'.\n"
        )

        messages = [
            {"role": "system", "content": SKILL_VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        validation_result: SkillValidationResult | None = None

        for step in range(self.max_steps):
            response_msg = self._call_llm(messages)
            assistant_msg = response_to_dict(response_msg)
            messages.append(assistant_msg)

            tool_calls = response_msg.tool_calls or []
            if not tool_calls:
                remaining = self.max_steps - step - 1
                messages.append({
                    "role": "user",
                    "content": f"Please continue validating the skill. ({remaining} steps remaining)",
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
                    v_result = tool_args.get("validation_result", "FAIL")
                    v_details = tool_args.get("validation_details", "")
                    validation_result = SkillValidationResult(
                        passed=(v_result == "PASS"),
                        details=v_details,
                    )
                    finished = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Validation complete: {v_result} — {v_details}",
                    })
                elif tool_name == "bash":
                    command = tool_args.get("command", "")
                    if not command:
                        observation = "Error: empty command"
                    else:
                        try:
                            r = workspace.execute_command(command, timeout=60)
                            if r.timeout_occurred:
                                observation = f"Error: Command timed out after 60s"
                            else:
                                parts = []
                                if r.stdout:
                                    parts.append(r.stdout)
                                if r.stderr:
                                    parts.append(r.stderr)
                                output = "\n".join(parts) or "(no output)"
                                if len(output) > 4000:
                                    output = output[:2000] + f"\n... ({len(output) - 4000} chars truncated) ...\n" + output[-2000:]
                                observation = output + f"\n[exit code: {r.exit_code}]"
                        except Exception as e:
                            observation = f"Error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Unknown tool '{tool_name}'",
                    })

            if finished:
                break

            remaining = self.max_steps - step - 1
            if 0 < remaining <= 3:
                messages.append({
                    "role": "user",
                    "content": f"[WARNING] You have only {remaining} step(s) remaining. Please call `finish` with your validation result.",
                })

        if validation_result is None:
            validation_result = SkillValidationResult(
                passed=False,
                details="Validation agent did not finish within max steps.",
            )

        return validation_result

    def _fix_skill(
        self,
        skill: SkillV3,
        validation_error: str,
    ) -> SkillV3 | None:
        prompt = SKILL_FIX_PROMPT.format(
            skill_name=skill.name,
            skill_description=skill.description,
            bash_script=skill.bash_script,
            py_script=skill.py_script,
            validation_error=validation_error,
        )

        messages = [
            {"role": "system", "content": "You are a skill fixer. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(3):
            try:
                response = self._caller.chat(messages=messages)
                content = (response or "").strip()
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
                    logger.info(f"[SkillValidator] Fix skipped: {reason}")
                    return None

                bash_script = parsed.get("bash_script", "")
                py_script = parsed.get("py_script", "")
                fix_desc = parsed.get("fix_description", "")

                if not bash_script or not py_script:
                    logger.warning(
                        f"[SkillValidator] Fix attempt {attempt + 1}: missing bash_script or py_script"
                    )
                    continue

                fixed_skill = SkillV3(
                    name=skill.name,
                    description=skill.description,
                    bash_script=bash_script,
                    py_script=py_script,
                    source_trajectory_summary=skill.source_trajectory_summary,
                )
                logger.info(
                    f"[SkillValidator] Fixed skill '{skill.name}': {fix_desc}"
                )
                return fixed_skill
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillValidator] Fix parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillValidator] Fix attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillValidator] Failed to fix skill after 3 attempts")
        return None

    def validate(
        self,
        skill: SkillV3,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
    ) -> SkillValidationResult:
        logger.info(f"[SkillValidator] Starting validation for skill: {skill.name}")

        snapshot_before = self._snapshot_workspace(workspace)

        current_skill = skill

        for fix_attempt in range(self.max_fix_attempts + 1):
            ok, err = self._deploy_skill_to_test_dir(current_skill, workspace)
            if not ok:
                self._cleanup(workspace)
                self._rollback_workspace(workspace, snapshot_before, self._snapshot_workspace(workspace))
                return SkillValidationResult(
                    passed=False,
                    details=err,
                    fixed_skill=current_skill if current_skill is not skill else None,
                )

            result = self._run_validation_agent(current_skill, workspace)

            if result.passed:
                self._cleanup(workspace)
                snapshot_after = self._snapshot_workspace(workspace)
                workspace_affected = snapshot_before != snapshot_after
                if workspace_affected:
                    logger.warning(
                        f"[SkillValidator] Workspace modified during validation of '{skill.name}'. Rolling back..."
                    )
                    self._rollback_workspace(workspace, snapshot_before, snapshot_after)

                result.workspace_affected = workspace_affected
                if current_skill is not skill:
                    result.fixed_skill = current_skill
                logger.info(
                    f"[SkillValidator] Skill '{skill.name}' PASSED on attempt {fix_attempt + 1}: {result.details}"
                )
                return result

            self._cleanup(workspace)

            if fix_attempt < self.max_fix_attempts:
                logger.info(
                    f"[SkillValidator] Skill '{skill.name}' FAILED on attempt {fix_attempt + 1}: {result.details}. "
                    f"Attempting fix ({fix_attempt + 1}/{self.max_fix_attempts})..."
                )
                fixed_skill = self._fix_skill(current_skill, result.details)
                if fixed_skill is None:
                    logger.info(f"[SkillValidator] Could not fix skill '{skill.name}'. Giving up.")
                    break
                current_skill = fixed_skill
            else:
                logger.info(
                    f"[SkillValidator] Skill '{skill.name}' FAILED after {fix_attempt + 1} attempts. No more fix attempts."
                )

        snapshot_after = self._snapshot_workspace(workspace)
        workspace_affected = snapshot_before != snapshot_after
        if workspace_affected:
            logger.warning(
                f"[SkillValidator] Workspace modified during validation of '{skill.name}'. Rolling back..."
            )
            self._rollback_workspace(workspace, snapshot_before, snapshot_after)

        result.workspace_affected = workspace_affected
        if current_skill is not skill:
            result.fixed_skill = current_skill
        logger.info(
            f"[SkillValidator] Skill '{skill.name}' validation FAILED: {result.details}"
        )
        return result

    @staticmethod
    def _cleanup(workspace: "DockerWorkspace") -> None:
        try:
            workspace.execute_command("rm -rf /workspace/.skill_test", timeout=10)
        except Exception as e:
            logger.warning(f"[SkillValidator] Cleanup failed: {e}")

    @staticmethod
    def _rollback_workspace(
        workspace: "DockerWorkspace",
        before: set[str],
        after: set[str],
    ) -> None:
        added = after - before
        for fpath in added:
            try:
                workspace.execute_command(f"rm -f '{fpath}'", timeout=5)
            except Exception:
                pass
