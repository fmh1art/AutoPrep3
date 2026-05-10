from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RuntimeInterventionV3:
    rule: str
    message: str
    revoke_step: bool = False


class RuntimeDetectorV3:
    """Rule-based, per-step cost / quality checks for the execution agent.

    Currently implemented rules:
      - R1 (duplicate tool call): if the same tool with exactly the same
        arguments has been invoked consecutively ``duplicate_k`` or more times,
        inject a user-role message asking the agent to stop repeating.
      - R2 (overlong observation): if the current step produced an observation
        longer than ``long_observation_threshold`` characters, revoke the step
        and inject a user-role advisory message.

    The detector keeps no side-channel state so that it is safe to share
    across subagents if desired.
    """

    def __init__(
        self,
        duplicate_k: int = 2,
        long_observation_threshold: int = 8000,
        long_observation_preview_chars: int = 800,
    ):
        if duplicate_k < 2:
            raise ValueError("duplicate_k must be >= 2 (k=1 would fire on every call)")
        self.duplicate_k = duplicate_k
        self.long_observation_threshold = long_observation_threshold
        self.long_observation_preview_chars = long_observation_preview_chars

    @staticmethod
    def _normalize_args(args: Any) -> str:
        try:
            return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return repr(args)

    def check_duplicate_tool_call(
        self,
        trajectory: list[dict],
    ) -> RuntimeInterventionV3 | None:
        tool_records = [r for r in trajectory if r.get("role") == "tool"]
        if len(tool_records) < self.duplicate_k:
            return None

        recent = tool_records[-self.duplicate_k:]
        first = recent[0]
        first_key = (first.get("tool_name"), self._normalize_args(first.get("tool_args", {})))

        for rec in recent[1:]:
            key = (rec.get("tool_name"), self._normalize_args(rec.get("tool_args", {})))
            if key != first_key:
                return None

        tool_name = first_key[0]
        args_preview = first_key[1]
        if len(args_preview) > 400:
            args_preview = args_preview[:400] + " …(truncated)"

        msg = (
            f"[Runtime Advisor] The tool `{tool_name}` has been called "
            f"{self.duplicate_k} consecutive times with exactly the same arguments "
            f"(args: {args_preview}). Repeating an identical call is almost certainly "
            f"wasted budget — please change your strategy (inspect the previous "
            f"observation, adjust arguments, or move on to a different step)."
        )
        return RuntimeInterventionV3(
            rule="duplicate_tool_call",
            message=msg,
            revoke_step=False,
        )

    def check_long_observation(
        self,
        tool_name: str,
        tool_args: dict,
        observation: str,
    ) -> RuntimeInterventionV3 | None:
        if observation is None:
            return None
        obs_len = len(observation)
        if obs_len < self.long_observation_threshold:
            return None

        try:
            args_repr = json.dumps(tool_args, ensure_ascii=False, indent=2, default=str)
        except Exception:
            args_repr = repr(tool_args)
        if len(args_repr) > 800:
            args_repr = args_repr[:800] + "\n...(truncated)"

        preview = observation[: self.long_observation_preview_chars]

        msg = (
            f"[Runtime Advisor] The tool call below produced an observation of "
            f"{obs_len} characters, exceeding the budget threshold of "
            f"{self.long_observation_threshold}. This step has been **revoked** "
            f"(no observation is being added to your context) to avoid blowing up "
            f"API cost.\n\n"
            f"Offending tool: `{tool_name}`\n"
            f"Arguments:\n```json\n{args_repr}\n```\n"
            f"Observation preview (first {self.long_observation_preview_chars} chars):\n"
            f"```\n{preview}\n```\n\n"
            f"Please retry with a *narrower* call — e.g. read a specific line range "
            f"(`sed -n 'A,Bp' file`), grep for specific patterns instead of dumping "
            f"the whole file, add `| head -n N` / `| tail -n N`, or target a single "
            f"directory / function rather than the entire repository."
        )
        return RuntimeInterventionV3(
            rule="long_observation",
            message=msg,
            revoke_step=True,
        )
