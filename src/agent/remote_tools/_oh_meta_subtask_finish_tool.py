"""Standalone SubtaskFinishTool definition (top-level module name).

This file MUST satisfy three constraints:

  1. It is referenced by the *exact same* qualname (``_oh_meta_subtask_finish_tool``)
     on both the host (client) side and inside the agent_server Docker container.
     OpenHands SDK records the qualname when ``register_tool()`` runs, then sends
     it to the server which calls ``importlib.import_module(qualname)``.

  2. It must NOT depend on any project-internal package
     (``src.agent.*``, ``src.benchmarks.*``, ``src.tools.*``, ``openhands.workspace``,
     etc.).  Inside the agent_server container only ``openhands.sdk`` /
     ``openhands.tools`` are guaranteed to exist.

  3. It is dropped into the container's ``site-packages`` directory at
     sub-agent startup so the server-side ``importlib.import_module`` call
     succeeds (see ``_install_subtask_finish_tool_in_container`` in
     ``meta_agent_openhands.py``).

The tool itself:
  - tool name (LLM-facing): ``finish``
  - parameters:
      * ``message`` (str, required) — concise summary of what was accomplished
      * ``useful_trajectory_indexes`` (list[int], required) — 0-based step
        indexes the next sub-agent should see.  Aim for 3-7 highly-selective
        steps.
  - on call: marks the conversation as FINISHED and echoes ``message`` as the
    observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


SUBTASK_FINISH_TOOL_DESCRIPTION = """Signal that the current SUBTASK is complete.

You MUST provide BOTH parameters as STRUCTURED tool arguments (not embedded in text):

  - `message`: a concise summary of what you accomplished (1-3 sentences).
  - `useful_trajectory_indexes`: a list of integer step indexes that contain
    information the NEXT sub-agent will need. Be HIGHLY selective:
      * Step indexes are 0-based — step 0 is the FIRST tool call you made.
      * Include ONLY steps that:
          1. Revealed the bug location (file path + line number)
          2. Showed the buggy code
          3. Performed the actual code change
          4. Verified the fix passed tests
      * EXCLUDE: ls/find/grep exploration that didn't lead anywhere, repeated
        reads, dead-end attempts.
      * Aim for 3-7 steps total. Including everything is WORSE than including
        nothing. If you genuinely had no useful steps, pass [].

Do NOT call this tool if you have not actually completed the subtask.
Do NOT embed the indexes in `message` — pass them as a real JSON array
through the `useful_trajectory_indexes` argument.
"""


class SubtaskFinishAction(Action):
    message: str = Field(
        description="Final summary of what this sub-agent accomplished. 1-3 sentences.",
    )
    useful_trajectory_indexes: list[int] = Field(
        description=(
            "REQUIRED. The 0-based indexes of trajectory steps that the next "
            "sub-agent MUST see. Include ONLY steps that revealed the bug, "
            "showed the buggy code, made the fix, or verified the fix. "
            "Exclude exploratory/dead-end steps. 3-7 steps typical."
        ),
    )


class SubtaskFinishObservation(Observation):
    message: str = Field(default="")
    useful_trajectory_indexes: list[int] = Field(default_factory=list)

    @property
    def to_llm_content(self):
        from openhands.sdk import TextContent
        return [TextContent(text=self.message or "Subtask finished.")]


class SubtaskFinishExecutor(ToolExecutor):
    def __call__(
        self,
        action: SubtaskFinishAction,
        conversation: Any = None,
    ) -> SubtaskFinishObservation:
        # Mark the conversation as finished so the run loop terminates.
        if conversation is not None:
            try:
                from openhands.sdk.conversation.state import (
                    ConversationExecutionStatus,
                )
                conversation.state.execution_status = (
                    ConversationExecutionStatus.FINISHED
                )
            except Exception:
                # If the SDK version differs slightly, fall back gracefully.
                pass
        return SubtaskFinishObservation(
            message=action.message,
            useful_trajectory_indexes=list(action.useful_trajectory_indexes or []),
        )


class SubtaskFinishTool(ToolDefinition[SubtaskFinishAction, SubtaskFinishObservation]):
    """Custom finish tool for sub-agents that captures useful_trajectory_indexes
    as a structured argument (not as a text-embedded XML tag)."""

    name: ClassVar[str] = "finish"

    @classmethod
    def create(
        cls,
        conv_state: Any = None,
        **params,
    ) -> Sequence["SubtaskFinishTool"]:
        return [
            cls(
                action_type=SubtaskFinishAction,
                observation_type=SubtaskFinishObservation,
                description=SUBTASK_FINISH_TOOL_DESCRIPTION,
                executor=SubtaskFinishExecutor(),
                annotations=ToolAnnotations(
                    title="finish",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


# Register on import. Both host (when meta_agent_openhands.py imports this)
# and server (when agent_server importlib.import_module's it) execute this.
register_tool("SubtaskFinishTool", SubtaskFinishTool)
