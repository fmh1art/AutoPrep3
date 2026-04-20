from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import yaml

from src.agent.code_agent import CodeAgent, AgentResult
from src.agent.operator import Operator, OperatorPlan, LLMBackbone, CognitiveType
from src.tools.funcs import render_j2, parse_any_string

logger = logging.getLogger(__name__)

PLANNING_SYSTEM_PROMPT = """\
You are a planning agent. Your job is to explore the codebase using the available \
tools (bash and file_editor), understand the task, and then output a decomposition \
plan as a sequence of operators.

You should use tools to:
- Explore the repository structure (ls, find, grep)
- Read relevant source files (file_editor view)
- Understand the codebase before planning

When you have enough understanding, call the `finish` tool with your plan in the message.

IMPORTANT RULES:
- Always use absolute file paths (starting with /).
- Explore the codebase BEFORE creating the plan — do not plan blindly.
- Focus on understanding the task requirements and code structure.
- Do NOT make any code changes — you are only planning, not implementing.
"""

PLANNING_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this tool when you have explored the codebase and are ready to output "
            "your operator plan. The message must contain the plan as a JSON code block."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Your operator plan as a JSON code block (```json ... ```). "
                        "Each operator must have 'index' and 'subtask' fields. "
                        "Example: ```json\\n"
                        '[\\n'
                        '  {"index": 1, "subtask": "Explore and locate the bug"},\\n'
                        '  {"index": 2, "subtask": "Implement the fix"},\\n'
                        '  {"index": 3, "subtask": "Verify the fix with tests"}\\n'
                        ']\\n'
                        "```"
                    ),
                },
            },
            "required": ["message"],
        },
    },
}


class OperatorPlanningAgent:
    def __init__(self, planner_cfg: dict, max_steps: int = 15):
        self.planner_cfg = planner_cfg
        self.max_steps = max_steps
        self._agent = CodeAgent(
            llm_cfg=planner_cfg,
            max_steps=max_steps,
            max_retries_per_call=3,
            system_prompt=PLANNING_SYSTEM_PROMPT,
            tool_definitions=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": (
                            "Execute a bash command in the terminal. "
                            "Use `&&` or `;` to chain multiple commands. "
                            "Commands run inside the Docker workspace."
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "file_editor",
                        "description": (
                            "Custom editing tool for viewing, creating and editing files.\n"
                            "Commands: view, create, str_replace, insert, undo_edit.\n"
                            "- view: display file content (cat -n) or list directory\n"
                            "- create: create a new file (fails if file exists)\n"
                            "- str_replace: replace exact string in file\n"
                            "- insert: insert text after a given line number\n"
                            "- undo_edit: revert last edit to a file\n\n"
                            "CRITICAL: old_str must match EXACTLY (whitespace included) and be unique."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                                    "description": "The file editor command.",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "Absolute path to file or directory.",
                                },
                                "file_text": {
                                    "type": "string",
                                    "description": "Required for `create`: content of the new file.",
                                },
                                "old_str": {
                                    "type": "string",
                                    "description": "Required for `str_replace`: exact string to find.",
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": "For `str_replace`: replacement string. For `insert`: string to insert.",
                                },
                                "insert_line": {
                                    "type": "integer",
                                    "description": "Required for `insert`: insert new_str AFTER this line number (0-indexed).",
                                },
                                "view_range": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Optional for `view`: [start_line, end_line]. Indexing starts at 1. Use -1 for end.",
                                },
                            },
                            "required": ["command", "path"],
                        },
                    },
                },
                PLANNING_FINISH_TOOL,
            ],
        )

    @property
    def caller(self):
        return self._agent.caller

    def _get_llm_backbone(self) -> str:
        my_llm_name = self.planner_cfg.get("llm_name", "")
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "_config",
        )
        mapping = {"CHEAP": "doubao.yaml", "EXPENSIVE": "kimi2.5.yaml"}
        for backbone_key, filename in mapping.items():
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                try:
                    cfg = yaml.safe_load(open(path, "r"))
                    if cfg.get("llm_name") == my_llm_name:
                        return backbone_key
                except Exception:
                    pass
        if "kimi" in my_llm_name.lower():
            return "EXPENSIVE"
        return "CHEAP"

    def generate_plan(
        self,
        instruction: str,
        workspace_overview: str = "",
        max_attempts: int = 3,
        workspace=None,
        repo_path: str = "/workspace",
    ) -> tuple[OperatorPlan, dict[str, Any]]:
        usage_before = self.caller.get_total_usage()

        prompt = render_j2(
            "operator_planning.j2",
            context={
                "instruction": instruction,
                "workspace_overview": workspace_overview,
            },
        )

        if random.random() < 0.1:
            logger.info(f"Operator planning prompt: {prompt[:2000]}")

        for attempt in range(1, max_attempts + 1):
            try:
                if workspace is not None:
                    result: AgentResult = self._agent.run(
                        instruction=prompt,
                        workspace=workspace,
                        output_dir=None,
                    )
                    raw_response = ""
                    if result.other_content.get("finish_message"):
                        raw_response = result.other_content["finish_message"]
                    if not raw_response and result.messages:
                        for msg in reversed(result.messages):
                            if msg.get("role") == "assistant" and msg.get("content"):
                                raw_response = msg["content"]
                                break
                else:
                    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
                    raw_response = self.caller.chat(messages)

                plan = self._parse_plan(raw_response)
                if plan is not None and len(plan.operators) > 0:
                    plan.reindex()
                    logger.info(
                        f"Operator plan generated: {len(plan.operators)} operators, "
                        f"backbone distribution: {self._backbone_distribution(plan)}"
                    )
                    metrics = self._compute_metrics(usage_before)
                    return plan, metrics
            except Exception as e:
                logger.error(f"Planning attempt {attempt} failed: {e}")

        logger.warning("All planning attempts failed, using fallback plan")
        metrics = self._compute_metrics(usage_before)
        return self._fallback_plan(), metrics

    def _compute_metrics(self, usage_before: dict) -> dict[str, Any]:
        usage_after = self.caller.get_total_usage()
        return {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "accumulated_cost": 0.0,
            "llm_backbone": self._get_llm_backbone(),
        }

    def _parse_plan(self, text: str) -> OperatorPlan | None:
        if not text:
            return None

        import re
        pattern = r"```json\s*(\[.*?\])\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid = all(
                        isinstance(item, dict)
                        and "subtask" in item
                        for item in parsed
                    )
                    if valid:
                        return OperatorPlan.from_dict_list(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                return OperatorPlan.from_dict_list(parsed)
        except Exception:
            pass

        return None

    @staticmethod
    def _backbone_distribution(plan: OperatorPlan) -> dict[str, int]:
        dist = {"CHEAP": 0, "EXPENSIVE": 0, "unassigned": 0}
        for op in plan.operators:
            if op.llm_backbone is not None:
                dist[op.llm_backbone.value] += 1
            else:
                dist["unassigned"] += 1
        return dist

    @staticmethod
    def _fallback_plan() -> OperatorPlan:
        return OperatorPlan(operators=[
            Operator(
                index=1,
                subtask="Explore the codebase to locate relevant files and understand the code structure",
                cognitive_type=CognitiveType.PERCEPTION,
                requires_info=[],
                produces_info=["relevant file paths", "code structure understanding"],
            ),
            Operator(
                index=2,
                subtask="Analyze the root cause and design the fix strategy",
                cognitive_type=CognitiveType.REASONING,
                requires_info=["relevant file paths", "code structure understanding"],
                produces_info=["root cause", "fix strategy"],
            ),
            Operator(
                index=3,
                subtask="Implement the required changes to solve the task",
                cognitive_type=CognitiveType.ACTION,
                requires_info=["root cause", "fix strategy"],
                produces_info=["modified files"],
            ),
            Operator(
                index=4,
                subtask="Run existing tests to verify the changes work correctly",
                cognitive_type=CognitiveType.PERCEPTION,
                requires_info=["modified files"],
                produces_info=["test results"],
            ),
        ])
