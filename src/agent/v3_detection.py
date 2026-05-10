from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from src.agent.v3_skills import SkillV3
from src.module.gpt_inference import SimpleAPICaller

logger = logging.getLogger(__name__)


@dataclass
class CostAnomalyV3:
    subtask_index: int
    subtask_description: str
    total_steps: int
    threshold: int
    trajectory_summary: str


class CostDetectorV3:
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
    ) -> CostAnomalyV3 | None:
        total_steps = len([r for r in trajectory_records if r.get("role") == "tool"])
        total_tokens = metrics.get("total_tokens", 0)

        if total_steps >= self.step_threshold or total_tokens >= self.token_threshold:
            summary = self._summarize_trajectory(trajectory_records)
            return CostAnomalyV3(
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


SKILL_NEED_DECISION_PROMPT = """You are a skill need analyzer. A subtask consumed too many steps ({total_steps} steps, threshold is {threshold}), indicating a cost anomaly.

## Subtask Description
{subtask_description}

## Trajectory Summary
{trajectory_summary}

## Your Task

Analyze the trajectory and determine whether the costly operation can be abstracted into a **reusable skill tool**. A skill is a parameterized bash+python script that encapsulates a multi-step workflow.

## Skill Requirements (ALL must be met to justify creating a skill)

1. **Reusable across projects** — solves a CLASS of problems, not one specific task. Would be useful for 5+ different tasks.
2. **Fully parameterized** — all paths, patterns, strings via CLI args. NO hardcoded task-specific values.
3. **Multi-step workflow** — encapsulates 3+ steps. Single-command wrappers are NOT valid skills.
4. **Self-contained** — works in a fresh Docker workspace with only stdlib packages.
5. **Clear contract** — description must include: usage syntax, parameter descriptions, output format, and a concrete example.
6. **Dependency self-contained** — if the skill depends on any third-party Python packages (non-stdlib), the bash_script MUST install them via `pip install` before running the Python script.

## Anti-Patterns (do NOT create a skill if the pattern matches any of these)
- Task-specific (e.g., `fix_auth_bug.sh`)
- Hardcoded paths (e.g., searches `/workspace/django/auth/`)
- Single-command alias (e.g., just wraps `pytest`)
- Too narrow (e.g., `find_todo_in_readme.sh`)

## Decision Criteria

- Look at the trajectory: is there a **repeated multi-step pattern** that could be encapsulated?
- Is the pattern **generic enough** to be reused across different projects/tasks?
- Would a skill **genuinely save steps** for similar future tasks?
- If the cost is due to task-specific debugging, trial-and-error, or one-off exploration, do NOT create a skill.

## Output Format

Respond with JSON only:
```json
{{
  "need_skill": true | false,
  "reason": "Brief explanation of your decision",
  "skill_purpose": "If need_skill is true, describe what the skill should do in one sentence. Otherwise null."
}}
```"""


@dataclass
class SkillNeedDecision:
    need_skill: bool
    reason: str
    skill_purpose: str | None = None


class SkillNeedDecider:
    def __init__(self, llm_cfg: dict):
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    def decide(self, anomaly: CostAnomalyV3) -> SkillNeedDecision:
        prompt = SKILL_NEED_DECISION_PROMPT.format(
            total_steps=anomaly.total_steps,
            threshold=anomaly.threshold,
            subtask_description=anomaly.subtask_description,
            trajectory_summary=anomaly.trajectory_summary[:8000],
        )

        messages = [
            {"role": "system", "content": "You are a skill need analyzer. Respond with valid JSON only."},
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
                need_skill = bool(parsed.get("need_skill", False))
                reason = parsed.get("reason", "")
                skill_purpose = parsed.get("skill_purpose") if need_skill else None

                logger.info(
                    f"[SkillNeedDecider] Decision: need_skill={need_skill}, reason={reason}"
                )
                return SkillNeedDecision(
                    need_skill=need_skill,
                    reason=reason,
                    skill_purpose=skill_purpose,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillNeedDecider] Parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillNeedDecider] Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillNeedDecider] Failed after 3 attempts, defaulting to no skill")
        return SkillNeedDecision(
            need_skill=False,
            reason="Failed to get LLM decision after 3 attempts",
        )


SKILL_EVOLUTION_PROMPT_V3 = """You are a skill designer. A subtask consumed too many steps ({total_steps} steps, threshold is {threshold}). Here is the trajectory:

{trajectory_summary}

Analyze the trajectory. If there is a **genuinely reusable, multi-step pattern**, create a skill. Otherwise, skip.

## Skill Requirements (ALL must be met, otherwise skip)

1. **Reusable across projects** — solves a CLASS of problems, not one specific task. Would be useful for 5+ different tasks.
2. **Fully parameterized** — all paths, patterns, strings via CLI args. NO hardcoded task-specific values.
3. **Multi-step workflow** — encapsulates 3+ steps. Single-command wrappers are NOT valid skills.
4. **Self-contained** — works in a fresh Docker workspace with only stdlib packages.
5. **Clear contract** — description must include: usage syntax, parameter descriptions, output format, and a concrete example.
6. **Dependency self-contained** — if the skill depends on any third-party Python packages (non-stdlib), the bash_script MUST install them via `pip install` before running the Python script. This ensures that after deploying the skill and running the bash command, the skill works immediately without any manual setup.

## Anti-Patterns (skip if the pattern matches any of these)
- Task-specific (e.g., `fix_auth_bug.sh`)
- Hardcoded paths (e.g., searches `/workspace/django/auth/`)
- Single-command alias (e.g., just wraps `pytest`)
- Too narrow (e.g., `find_todo_in_readme.sh`)

## Output Format

Create a skill:
```json
{{
  "name": "skill_name",
  "description": "skill_name.sh <param1> <param2> [optional_param]\\n\\n<What it does>\\n\\nParameters:\\n  param1: <description>\\n  param2: <description>\\n  optional_param: <description, default>\\n\\nOutput: <what is printed on success>\\n\\nExample: bash /workspace/.skill/skill_name.sh 'def run' /workspace/src content",
  "bash_script": "#!/bin/bash\\nset -e\\npip install -q <required_packages> 2>/dev/null\\npython3 /workspace/.skill/skill_name.py \"$@\"\\n",
  "py_script": "#!/usr/bin/env python3\\nimport sys, argparse\\n..."
}}
```

Skip (no valid skill):
```json
{{"skip": true, "reason": "explanation"}}
```

Notes: Scripts go to `/workspace/.skill/`. Use absolute paths. Handle errors gracefully. If the skill depends on any third-party packages, the bash_script MUST include `pip install <package>` before calling the Python script, so the skill is ready to run immediately after deployment.
"""


class SkillEvolverV3:
    def __init__(self, llm_cfg: dict):
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    def evolve(self, anomaly: CostAnomalyV3) -> tuple[SkillV3 | None, str]:
        prompt = SKILL_EVOLUTION_PROMPT_V3.format(
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
                    logger.info(f"[SkillEvolverV3] Skipped: {reason}")
                    return None, reason

                name = parsed.get("name", "")
                description = parsed.get("description", "")
                bash_script = parsed.get("bash_script", "")
                py_script = parsed.get("py_script", "")

                if not name or not description or not bash_script or not py_script:
                    missing = [k for k in ("name", "description", "bash_script", "py_script") if not parsed.get(k)]
                    logger.warning(
                        f"[SkillEvolverV3] Skill missing required fields: {missing}. Skipping."
                    )
                    return None, f"Skill missing required fields: {missing}"

                skill = SkillV3(
                    name=name,
                    description=description,
                    bash_script=bash_script,
                    py_script=py_script,
                    source_trajectory_summary=anomaly.trajectory_summary[:500],
                )
                logger.info(f"[SkillEvolverV3] Generated skill: {skill.name}")
                return skill, ""
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillEvolverV3] Parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillEvolverV3] Attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillEvolverV3] Failed to generate skill after 3 attempts")
        return None, "failed to parse LLM response after 3 attempts"
