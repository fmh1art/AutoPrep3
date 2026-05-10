from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from src.agent.v3_skills import SkillV3, SkillRegistryV3
from src.module.gpt_inference import SimpleAPICaller

logger = logging.getLogger(__name__)


SKILL_EVALUATION_PROMPT = """You are a skill hub manager. Your job is to decide whether a newly generated skill should be added to the skill hub.

## Existing Skills in Hub
{existing_skills_section}

## New Skill to Evaluate
- **Name**: {new_skill_name}
- **Description**: {new_skill_description}

## Your Task

Compare the new skill with existing skills and decide one of the following actions:

1. **"add"** — The new skill provides functionality NOT covered by any existing skill. It is genuinely useful and reusable.
2. **"skip"** — A similar or equivalent skill already exists in the hub, and the new one does NOT offer meaningful improvements.
3. **"replace"** — The new skill is significantly better than an existing similar skill (e.g., more general, better error handling, covers more cases, more efficient). The old skill should be replaced.

## Decision Criteria

- Two skills are "similar" if they solve the same class of problems or can be used interchangeably in most scenarios.
- A new skill is "better" if it:
  - Covers a superset of the old skill's functionality
  - Has better error handling or robustness
  - Is more general-purpose (less task-specific)
  - Produces more useful output
- Do NOT add a skill that is too narrow or task-specific if a broader skill already covers it.
- Do NOT replace a skill unless the new one is clearly superior — prefer stability.

## Output Format

Respond with JSON only:
```json
{{
  "action": "add" | "skip" | "replace",
  "reason": "Brief explanation of your decision",
  "replace_target": "name of the existing skill to replace (only if action is replace, otherwise null)"
}}
```"""

SKILL_SELECTION_PROMPT = """You are a skill hub manager. Given a subtask that a coding agent needs to execute, select which skills from the hub are relevant and might help the agent complete the subtask more efficiently.

## Subtask Description
{subtask_description}

## Available Skills in Hub
{skill_list_section}

## Your Task

Select the skills that are most likely to be useful for this subtask. Consider:
- What operations does the subtask require? (e.g., searching code, editing files, running tests, analyzing logs)
- Which skills can help perform those operations more efficiently than manual bash commands?
- A skill is relevant if it could reasonably save steps for this type of task
- It is better to include a marginally relevant skill than to miss a useful one
- However, do NOT include skills that are clearly unrelated to the subtask

## Output Format

Respond with JSON only:
```json
{{
  "selected_skills": ["skill_name_1", "skill_name_2", ...],
  "reasoning": "Brief explanation of why these skills were selected"
}}
```"""


@dataclass
class SkillHubEvalResult:
    action: str
    reason: str
    replace_target: str | None = None


class SkillHubManager:
    def __init__(
        self,
        llm_cfg: dict,
        registry: SkillRegistryV3 | None = None,
        selection_threshold: int = 3,
    ):
        self._registry = registry or SkillRegistryV3()
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )
        self.selection_threshold = selection_threshold

    @property
    def registry(self) -> SkillRegistryV3:
        return self._registry

    def evaluate_and_register(self, new_skill: SkillV3) -> SkillHubEvalResult:
        existing = [s for s in self._registry.all_skills() if s.bash_script and s.py_script]
        if not existing:
            self._registry.register(new_skill)
            logger.info(f"[SkillHubManager] No existing skills — auto-registered: {new_skill.name}")
            return SkillHubEvalResult(action="add", reason="No existing skills in hub")

        eval_result = self._llm_evaluate(new_skill, existing)
        if eval_result.action == "add":
            self._registry.register(new_skill)
            logger.info(f"[SkillHubManager] Added new skill: {new_skill.name} — {eval_result.reason}")
        elif eval_result.action == "skip":
            logger.info(f"[SkillHubManager] Skipped skill: {new_skill.name} — {eval_result.reason}")
        elif eval_result.action == "replace":
            target = eval_result.replace_target
            if target and self._registry.get(target):
                if target != new_skill.name:
                    del self._registry._skills[target]
                    logger.info(f"[SkillHubManager] Removed old skill: {target}")
                self._registry.register(new_skill)
                logger.info(
                    f"[SkillHubManager] Replaced skill '{target}' with '{new_skill.name}' — {eval_result.reason}"
                )
            else:
                self._registry.register(new_skill)
                logger.warning(
                    f"[SkillHubManager] Replace target '{target}' not found, adding as new: {new_skill.name}"
                )
                eval_result.action = "add"
        else:
            logger.warning(f"[SkillHubManager] Unknown action '{eval_result.action}', defaulting to add")
            self._registry.register(new_skill)
            eval_result.action = "add"

        return eval_result

    def _llm_evaluate(self, new_skill: SkillV3, existing_skills: list[SkillV3]) -> SkillHubEvalResult:
        existing_section = ""
        for s in existing_skills:
            existing_section += f"### {s.name}\n"
            existing_section += f"- **Script Path**: `/workspace/.skill/{s.name}.sh`\n"
            existing_section += f"- **Entry Command**: `bash /workspace/.skill/{s.name}.sh <args>`\n"
            existing_section += f"- **Description**: {s.description}\n\n"
        if not existing_section:
            existing_section = "(none)"

        prompt = SKILL_EVALUATION_PROMPT.format(
            existing_skills_section=existing_section,
            new_skill_name=new_skill.name,
            new_skill_description=new_skill.description,
        )
        messages = [
            {"role": "system", "content": "You are a skill hub manager. Respond with valid JSON only."},
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
                action = parsed.get("action", "add")
                reason = parsed.get("reason", "")
                replace_target = parsed.get("replace_target")

                if action not in ("add", "skip", "replace"):
                    action = "add"

                return SkillHubEvalResult(
                    action=action,
                    reason=reason,
                    replace_target=replace_target if action == "replace" else None,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillHubManager] Eval parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillHubManager] Eval attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillHubManager] Eval failed after 3 attempts, defaulting to add")
        return SkillHubEvalResult(action="add", reason="LLM evaluation failed, defaulting to add")

    def select_relevant_skills(self, subtask_description: str) -> list[SkillV3]:
        all_skills = [s for s in self._registry.all_skills() if s.bash_script and s.py_script]
        if not all_skills:
            return []

        if len(all_skills) <= self.selection_threshold:
            logger.info(
                f"[SkillHubManager] {len(all_skills)} skills ≤ threshold "
                f"{self.selection_threshold}, returning all"
            )
            return all_skills

        selected_names = self._llm_select(subtask_description, all_skills)
        selected_skills = []
        for name in selected_names:
            skill = self._registry.get(name)
            if skill:
                selected_skills.append(skill)
            else:
                logger.warning(f"[SkillHubManager] LLM selected unknown skill: {name}")

        if not selected_skills:
            logger.info("[SkillHubManager] LLM selected no skills, falling back to all")
            return all_skills

        logger.info(
            f"[SkillHubManager] Selected {len(selected_skills)}/{len(all_skills)} skills "
            f"for subtask: {selected_names}"
        )
        return selected_skills

    def _llm_select(self, subtask_description: str, all_skills: list[SkillV3]) -> list[str]:
        skill_list_section = ""
        for s in all_skills:
            skill_list_section += f"### {s.name}\n"
            skill_list_section += f"- **Script Path**: `/workspace/.skill/{s.name}.sh`\n"
            skill_list_section += f"- **Entry Command**: `bash /workspace/.skill/{s.name}.sh <args>`\n"
            skill_list_section += f"- **Description**: {s.description}\n\n"

        prompt = SKILL_SELECTION_PROMPT.format(
            subtask_description=subtask_description[:2000],
            skill_list_section=skill_list_section,
        )
        messages = [
            {"role": "system", "content": "You are a skill hub manager. Respond with valid JSON only."},
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
                selected = parsed.get("selected_skills", [])
                if isinstance(selected, list):
                    return [str(s) for s in selected]
                return []
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[SkillHubManager] Select parse attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"[SkillHubManager] Select attempt {attempt + 1} failed: {e}")
                time.sleep(2)

        logger.warning("[SkillHubManager] Skill selection failed after 3 attempts, returning all")
        return [s.name for s in all_skills]

    def build_skill_prompt_for_subtask(self, subtask_description: str) -> str:
        selected = self.select_relevant_skills(subtask_description)
        if not selected:
            return ""

        parts = [
            "## Available Skills\n",
            "**IMPORTANT: Prefer using available skills over manual multi-step bash commands.** "
            "Skills encapsulate common multi-step operations and can save significant steps. "
            "Before writing a sequence of bash commands, check if an available skill already does what you need.\n",
        ]
        for skill in selected:
            parts.append(f"### {skill.name}\n")
            parts.append(f"- **Script Path**: `/workspace/.skill/{skill.name}.sh`\n")
            parts.append(f"- **Entry Command**: `bash /workspace/.skill/{skill.name}.sh <args>`\n")
            parts.append(f"- **Description**: {skill.description}\n")
        return "\n".join(parts)
