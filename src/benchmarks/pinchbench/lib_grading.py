"""
PinchBench grading engine - adapted from https://github.com/pinchbench/skill.git

This module provides grading functionality for PinchBench tasks.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lib_tasks import Task


logger = logging.getLogger(__name__)


DEFAULT_JUDGE_TIMEOUT_SECONDS = 300
MAX_TRANSCRIPT_CHARS = 8000
MAX_WORKSPACE_FILE_CHARS = 3000

_JUDGE_SYSTEM_MSG = (
    "You are a strict grading function. "
    "Respond with ONLY a JSON object, no prose, no markdown fences, no extra text."
)


def _get_judge_caller(judge_cfg: Dict[str, Any]):
    """Get a SimpleAPICaller instance for the judge model."""
    from src.module.gpt_inference import SimpleAPICaller
    
    return SimpleAPICaller(
        llm_name=judge_cfg.get("llm_name", ""),
        api_key=judge_cfg.get("key", ""),
        base_url=judge_cfg.get("openai_base_url", None),
    )


@dataclass
class GradeResult:
    task_id: str
    score: float
    max_score: float
    grading_type: str
    breakdown: Dict[str, float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "max_score": self.max_score,
            "grading_type": self.grading_type,
            "breakdown": self.breakdown,
            "notes": self.notes,
        }


def grade_task(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    skill_dir: Path,
    judge_cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> GradeResult:
    """
    Grade a task execution result.

    Args:
        task: The Task object
        execution_result: Dict containing 'transcript' and 'workspace' keys
        skill_dir: Path to the skill benchmark root directory
        judge_cfg: Judge model configuration (llm_name, key, openai_base_url)
        verbose: Enable verbose logging

    Returns:
        GradeResult with score and breakdown
    """
    grading_type = task.grading_type
    if verbose:
        logger.info("   Grading task %s with type: %s", task.task_id, grading_type)
        logger.info("   Execution status: %s", execution_result.get("status", "unknown"))

    if grading_type == "automated":
        result = _grade_automated(task, execution_result, skill_dir=skill_dir, verbose=verbose)
        if verbose:
            logger.info("   Automated grade breakdown: %s", result.breakdown)
        return result
    if grading_type == "llm_judge":
        result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            skill_dir=skill_dir,
            judge_cfg=judge_cfg,
            verbose=verbose,
        )
        if verbose:
            logger.info("   LLM judge breakdown: %s", result.breakdown)
        return result
    if grading_type == "hybrid":
        auto_result = _grade_automated(task, execution_result, skill_dir=skill_dir, verbose=verbose)
        llm_result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            skill_dir=skill_dir,
            judge_cfg=judge_cfg,
            verbose=verbose,
        )
        return _combine_grades(task, auto_result, llm_result)
    raise ValueError(f"Unknown grading type: {grading_type}")


def _grade_automated(
    task: Task,
    execution_result: Dict[str, Any],
    skill_dir: Optional[Path] = None,
    verbose: bool = False,
) -> GradeResult:
    """Grade a task using automated checks."""
    grading_code = _extract_grading_code(task)
    if not grading_code:
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="No automated grading code found",
        )

    # Convert transcript to OpenClaw format for compatibility with grading code
    transcript = _convert_to_openclaw_format(execution_result.get("transcript", []))

    namespace: Dict[str, Any] = {}
    exec(grading_code, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="Automated grading function missing",
        )

    try:
        scores = grade_func(
            transcript,
            execution_result.get("workspace", ""),
        )
        if not isinstance(scores, dict):
            scores = {}
    except Exception as e:
        logger.error("Automated grading failed for %s: %s", task.task_id, e)
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes=f"Automated grading error: {e}",
        )

    if verbose:
        logger.info("   Automated grading scores: %s", scores)

    total = _average_scores(scores)
    return GradeResult(
        task_id=task.task_id,
        score=total,
        max_score=1.0,
        grading_type="automated",
        breakdown=_normalize_score_dict(scores),
        notes="",
    )


def _convert_to_openclaw_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert AutoPrep3 message format to OpenClaw transcript format.

    AutoPrep3 format (from messages.jsonl):
        {"role": "system", "content": "..."}
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "...", "content": "..."}

    OpenClaw format (expected by grading code):
        {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]}}
        {"type": "message", "message": {"role": "toolResult", "content": ["..."]}}
    """
    transcript: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            # System messages are usually not included in the transcript for grading
            continue

        if role == "user":
            transcript.append({
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": content if isinstance(content, str) else json.dumps(content)}],
                },
            })

        elif role == "assistant":
            content_items = []
            if isinstance(content, str) and content:
                content_items.append({"type": "text", "text": content})

            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                tc_name = tc.get("function", {}).get("name", "")
                tc_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args_dict = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                except json.JSONDecodeError:
                    args_dict = {"raw": tc_args}
                content_items.append({
                    "type": "toolCall",
                    "name": tc_name,
                    "arguments": args_dict,
                })

            transcript.append({
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": content_items,
                },
            })

        elif role == "tool":
            tool_content = content if isinstance(content, str) else json.dumps(content)
            transcript.append({
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": [tool_content],
                },
            })

    return transcript


def _summarize_transcript(messages: List[Dict[str, Any]], max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Summarize transcript for LLM judge, truncating if necessary."""
    lines: List[str] = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue

        if isinstance(content, str):
            content_text = content
        else:
            content_text = json.dumps(content, ensure_ascii=False)

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                tc_name = tc.get("function", {}).get("name", "")
                tc_args = tc.get("function", {}).get("arguments", "{}")
                try:
                    args_dict = json.loads(tc_args) if isinstance(tc_args, str) else tc_args
                    args_text = json.dumps(args_dict, ensure_ascii=False)
                except json.JSONDecodeError:
                    args_text = str(tc_args)
                line = f"[{role}] tool_call: {tc_name}({args_text})"
                if total_chars + len(line) > max_chars:
                    lines.append("... [truncated] ...")
                    return "\n".join(lines)
                lines.append(line)
                total_chars += len(line)

        if content_text:
            line = f"[{role}] {content_text}"
            if total_chars + len(line) > max_chars:
                lines.append("... [truncated] ...")
                return "\n".join(lines)
            lines.append(line)
            total_chars += len(line)

    return "\n".join(lines)


def _read_workspace_files(workspace_path: str, max_chars: int = MAX_WORKSPACE_FILE_CHARS) -> str:
    """Read key files from workspace for LLM judge."""
    workspace = Path(workspace_path)
    if not workspace.exists():
        return ""

    output: List[str] = []
    total_chars = 0

    # Read common file types
    patterns = ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.csv"]
    for pattern in patterns:
        for file_path in sorted(workspace.rglob(pattern)):
            if file_path.is_dir():
                continue
            # Skip hidden files and large files
            if file_path.name.startswith(".") or file_path.stat().st_size > max_chars:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) == 0:
                    continue
                rel_path = file_path.relative_to(workspace)
                header = f"--- File: {rel_path} ---"
                if total_chars + len(header) + len(content) > max_chars * 3:
                    output.append("... [more files omitted] ...")
                    return "\n".join(output)
                output.append(header)
                output.append(content[:max_chars])
                total_chars += len(header) + len(content)
            except Exception:
                continue

    return "\n".join(output)


def _build_judge_prompt(
    task: Task,
    transcript_summary: str,
    workspace_content: str,
) -> str:
    """Build the LLM judge prompt."""
    criteria_text = "\n".join(f"- {c}" for c in task.grading_criteria) if task.grading_criteria else "N/A"

    rubric_text = task.llm_judge_rubric or "Evaluate based on the grading criteria above."

    prompt_parts = [
        "## TASK DESCRIPTION",
        f"**Task:** {task.name}",
        f"**Category:** {task.category}",
        "",
        "### User Request",
        task.prompt,
        "",
        "### Expected Behavior",
        task.expected_behavior or "N/A",
        "",
        "## GRADING CRITERIA",
        "Evaluate the agent's work based on these criteria. Each criterion is worth 1 point:",
        criteria_text,
        "",
        "## EVALUATION RUBRIC",
        rubric_text,
        "",
        "## AGENT TRANSCRIPT",
        transcript_summary or "(No transcript available)",
        "",
    ]

    if workspace_content:
        prompt_parts.extend([
            "## WORKSPACE FILES",
            workspace_content,
            "",
        ])

    prompt_parts.extend([
        "## INSTRUCTIONS",
        "You are a strict, objective judge. Evaluate the agent's work against the grading criteria.",
        "",
        "For each criterion, assign a score of 1.0 if fully met, 0.5 if partially met, or 0.0 if not met.",
        "",
        "Respond with ONLY a JSON object in this exact format (no extra text, no markdown):",
        "{",
        '  "scores": {',
        '    "criterion_1": 1.0,',
        '    "criterion_2": 0.5,',
        '    "criterion_3": 0.0',
        "  },",
        '  "notes": "Brief explanation of scoring decisions"',
        "}",
        "",
        "IMPORTANT:",
        "- The keys in 'scores' MUST match the grading criteria text exactly",
        "- Each score must be 0.0, 0.5, or 1.0",
        "- Do NOT include any text before or after the JSON object",
        "- Do NOT wrap the JSON in markdown code fences",
    ])

    return "\n".join(prompt_parts)


def _parse_judge_response(response_text: str, task: Task) -> Dict[str, Any]:
    """Parse the LLM judge response JSON."""
    # Try to extract JSON from the response
    text = response_text.strip()

    if not text:
        raise ValueError("Empty judge response")

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try to find JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    if not text:
        raise ValueError(f"No JSON found in judge response: {response_text[:500]}")

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse judge response as JSON: {e}\nResponse: {response_text[:500]}")

    if not isinstance(result, dict):
        raise ValueError(f"Judge response is not a JSON object: {result}")

    scores = result.get("scores", {})
    if not isinstance(scores, dict):
        raise ValueError(f"'scores' field is not a dict: {scores}")

    # Normalize scores
    normalized_scores: Dict[str, float] = {}
    for criterion, score in scores.items():
        try:
            normalized_scores[criterion] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            normalized_scores[criterion] = 0.0

    # If criteria names don't match, try to match by position
    if task.grading_criteria and len(normalized_scores) != len(task.grading_criteria):
        logger.warning(
            "   Judge scores count (%d) doesn't match criteria count (%d) for %s",
            len(normalized_scores), len(task.grading_criteria), task.task_id,
        )

    notes = result.get("notes", "")
    if not isinstance(notes, str):
        notes = str(notes)

    return {"scores": normalized_scores, "notes": notes}


def _grade_llm_judge(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    skill_dir: Optional[Path] = None,
    judge_cfg: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> GradeResult:
    """
    Grade a task using LLM judge.

    Args:
        task: The Task object
        execution_result: Dict containing 'transcript' and 'workspace' keys
        skill_dir: Path to the skill benchmark root directory
        judge_cfg: Judge model configuration (llm_name, key, openai_base_url)
        verbose: Enable verbose logging

    Returns:
        GradeResult with score and breakdown
    """
    task_id = task.task_id
    transcript = execution_result.get("transcript", [])
    execution_status = execution_result.get("status", "unknown")
    workspace_path = execution_result.get("workspace", "")

    if not transcript and execution_status != "success":
        if verbose:
            logger.info(
                "   Skipping LLM judge: status=%s, transcript empty",
                execution_status,
            )
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes=f"Skipped: task execution failed ({execution_status}), no transcript to evaluate",
        )

    if judge_cfg is None:
        logger.warning(
            "   No judge_cfg provided for LLM judge on %s. Scoring 0.0.",
            task_id,
        )
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes="No judge configuration provided",
        )

    # Prepare inputs for judge
    transcript_summary = _summarize_transcript(transcript)
    workspace_content = _read_workspace_files(workspace_path)

    judge_prompt = _build_judge_prompt(task, transcript_summary, workspace_content)

    if verbose:
        logger.info(
            "   LLM judge prompt length: %d chars, transcript: %d chars, workspace: %d chars",
            len(judge_prompt),
            len(transcript_summary),
            len(workspace_content),
        )

    # Call judge model
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            caller = _get_judge_caller(judge_cfg)

            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_MSG},
                {"role": "user", "content": judge_prompt},
            ]

            response = caller.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                timeout=DEFAULT_JUDGE_TIMEOUT_SECONDS,
            )

            response_text = response.get("content", "")
            if not response_text:
                raise ValueError("Judge returned empty response")

            if verbose:
                logger.info("   Judge response (attempt %d): %s", attempt + 1, response_text[:200])

            # Parse response
            parsed = _parse_judge_response(response_text, task)
            scores = parsed["scores"]
            notes = parsed["notes"]

            # Calculate total score
            if scores:
                total_score = sum(scores.values()) / len(scores)
            else:
                total_score = 0.0

            # Log usage
            usage = caller.get_total_usage()
            if verbose:
                logger.info(
                    "   Judge usage: input=%d, output=%d, cached=%d",
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cached_tokens"],
                )

            return GradeResult(
                task_id=task_id,
                score=total_score,
                max_score=1.0,
                grading_type="llm_judge",
                breakdown=scores,
                notes=notes,
            )

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "   LLM judge attempt %d/%d failed for %s: %s",
                attempt + 1, max_retries, task_id, e,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    # All retries failed
    logger.error(
        "   LLM judge failed after %d attempts for %s: %s",
        max_retries, task_id, last_error,
    )
    return GradeResult(
        task_id=task_id,
        score=0.0,
        max_score=1.0,
        grading_type="llm_judge",
        breakdown={},
        notes=f"LLM judge failed after {max_retries} attempts: {last_error}",
    )


def _combine_grades(task: Task, auto_result: GradeResult, llm_result: GradeResult) -> GradeResult:
    """Combine automated and LLM judge grades."""
    weights = task.grading_weights or {"automated": 0.5, "llm_judge": 0.5}
    auto_weight = float(weights.get("automated", 0.5))
    llm_weight = float(weights.get("llm_judge", 0.5))
    total_weight = auto_weight + llm_weight
    if total_weight <= 0:
        auto_weight = llm_weight = 0.5
        total_weight = 1.0
    combined_score = (
        auto_result.score * auto_weight + llm_result.score * llm_weight
    ) / total_weight
    breakdown = {
        **{f"automated.{k}": v for k, v in auto_result.breakdown.items()},
        **{f"llm_judge.{k}": v for k, v in llm_result.breakdown.items()},
    }
    notes = " | ".join(filter(None, [auto_result.notes, llm_result.notes]))
    return GradeResult(
        task_id=task.task_id,
        score=combined_score,
        max_score=1.0,
        grading_type="hybrid",
        breakdown=breakdown,
        notes=notes,
    )


def _extract_grading_code(task: Task) -> str:
    """Extract Python grading code from task's automated checks section."""
    if not task.automated_checks:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", task.automated_checks, re.DOTALL)
    if not match:
        return ""
    return match.group(1)


def _average_scores(scores: Dict[str, Any]) -> float:
    """Calculate average of scores."""
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_score_dict(scores: Dict[str, Any]) -> Dict[str, float]:
    """Normalize score dict values to floats."""
    normalized: Dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized
