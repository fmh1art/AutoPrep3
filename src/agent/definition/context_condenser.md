> ## Documentation Index
> Fetch the complete documentation index at: https://docs.openhands.dev/llms.txt
> Use this file to discover all available pages before exploring further.

# Context Condenser

> Manage agent memory by condensing conversation history to save tokens.

export const path_to_script_0 = "examples/01_standalone_sdk/14_context_condenser.py"

> A ready-to-run example is available [here](#ready-to-run-example)!

## What is a Context Condenser?

A **context condenser** is a crucial component that addresses one of the most persistent challenges in AI agent development: managing growing conversation context efficiently. As conversations with AI agents grow longer, the cumulative history leads to:

* **💰 Increased API Costs**: More tokens in the context means higher costs per API call
* **⏱️ Slower Response Times**: Larger contexts take longer to process
* **📉 Reduced Effectiveness**: LLMs become less effective when dealing with excessive irrelevant information

The context condenser solves this by intelligently summarizing older parts of the conversation while preserving essential information needed for the agent to continue working effectively.

## Default Implementation: `LLMSummarizingCondenser`

OpenHands SDK provides `LLMSummarizingCondenser` as the default condenser implementation. This condenser uses an LLM to generate summaries of conversation history when it exceeds the configured size limit.

### How It Works

When conversation history exceeds a defined threshold, the LLM-based condenser:

1. **Keeps recent messages intact** - The most recent exchanges remain unchanged for immediate context
2. **Preserves key information** - Important details like user goals, technical specifications, and critical files are retained
3. **Summarizes older content** - Earlier parts of the conversation are condensed into concise summaries using LLM-generated summaries
4. **Maintains continuity** - The agent retains awareness of past progress without processing every historical interaction

<img className="block dark:hidden" src="https://mintcdn.com/allhandsai/hwL6uO0ZqM-lmlYG/sdk/guides/assets/condenser_overview_light_mode.png?fit=max&auto=format&n=hwL6uO0ZqM-lmlYG&q=85&s=dc85afa9f2821b1f7a305ad1406ab3d9" alt="Light mode interface" width="3061" height="1677" data-path="sdk/guides/assets/condenser_overview_light_mode.png" />

<img className="hidden dark:block" src="https://mintcdn.com/allhandsai/hwL6uO0ZqM-lmlYG/sdk/guides/assets/condenser_overview_dark_mode.png?fit=max&auto=format&n=hwL6uO0ZqM-lmlYG&q=85&s=80dc8ab896f54e426af75b31ac159be0" alt="Dark mode interface" width="3061" height="1677" data-path="sdk/guides/assets/condenser_overview_dark_mode.png" />

This approach achieves remarkable efficiency gains:

* Up to **2x reduction** in per-turn API costs
* **Consistent response times** even in long sessions
* **Equivalent or better performance** on software engineering tasks

Learn more about the implementation and benchmarks in our [blog post on context condensation](https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents).

### Extensibility

The `LLMSummarizingCondenser` extends the `RollingCondenser` base class, which provides a framework for condensers that work with rolling conversation history. You can create custom condensers by extending base classes ([source code](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/base.py)):

* **`RollingCondenser`** - For condensers that apply condensation to rolling history
* **`CondenserBase`** - For more specialized condensation strategies

This architecture allows you to implement custom condensation logic tailored to your specific needs while leveraging the SDK's conversation management infrastructure.

### Setting Up Condensing

Create a `LLMSummarizingCondenser` to manage the context.
The condenser will automatically truncate conversation history when it exceeds max\_size, and replaces the dropped events with an LLM-generated summary.

This condenser triggers when there are more than `max_context_length` events in
the conversation history, and always keeps the first `keep_first` events (system prompts,
initial user messages) to preserve important context.

```python focus={3-4} icon="python" theme={null}
from openhands.sdk.context import LLMSummarizingCondenser

condenser = LLMSummarizingCondenser(
    llm=llm.model_copy(update={"usage_id": "condenser"}), max_size=10, keep_first=2
)

# Agent with condenser
agent = Agent(llm=llm, tools=tools, condenser=condenser)
```

### Ready-to-run example

<Note>
  This example is available on GitHub: [examples/01\_standalone\_sdk/14\_context\_condenser.py](https://github.com/OpenHands/software-agent-sdk/blob/main/examples/01_standalone_sdk/14_context_condenser.py)
</Note>

Automatically condense conversation history when context length exceeds limits, reducing token usage while preserving important information:

```python icon="python" expandable examples/01_standalone_sdk/14_context_condenser.py theme={null}
"""
To manage context in long-running conversations, the agent can use a context condenser
that keeps the conversation history within a specified size limit. This example
demonstrates using the `LLMSummarizingCondenser`, which automatically summarizes
older parts of the conversation when the history exceeds a defined threshold.
"""

import os

from pydantic import SecretStr

from openhands.sdk import (
    LLM,
    Agent,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

# Configure LLM
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
base_url = os.getenv("LLM_BASE_URL")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

# Tools
cwd = os.getcwd()
tools = [
    Tool(
        name=TerminalTool.name,
    ),
    Tool(name=FileEditorTool.name),
    Tool(name=TaskTrackerTool.name),
]

# Create a condenser to manage the context. The condenser will automatically truncate
# conversation history when it exceeds max_size, and replaces the dropped events with an
#  LLM-generated summary. This condenser triggers when there are more than ten events in
# the conversation history, and always keeps the first two events (system prompts,
# initial user messages) to preserve important context.
condenser = LLMSummarizingCondenser(
    llm=llm.model_copy(update={"usage_id": "condenser"}), max_size=10, keep_first=2
)

# Agent with condenser
agent = Agent(llm=llm, tools=tools, condenser=condenser)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent,
    callbacks=[conversation_callback],
    persistence_dir="./.conversations",
    workspace=".",
)

# Send multiple messages to demonstrate condensation
print("Sending multiple messages to demonstrate LLM Summarizing Condenser...")

conversation.send_message(
    "Hello! Can you create a Python file named math_utils.py with functions for "
    "basic arithmetic operations (add, subtract, multiply, divide)?"
)
conversation.run()

conversation.send_message(
    "Great! Now add a function to calculate the factorial of a number."
)
conversation.run()

conversation.send_message("Add a function to check if a number is prime.")
conversation.run()

conversation.send_message(
    "Add a function to calculate the greatest common divisor (GCD) of two numbers."
)
conversation.run()

conversation.send_message(
    "Now create a test file to verify all these functions work correctly."
)
conversation.run()

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")

# Conversation persistence
print("Serializing conversation...")

del conversation

# Deserialize the conversation
print("Deserializing conversation...")
conversation = Conversation(
    agent=agent,
    callbacks=[conversation_callback],
    persistence_dir="./.conversations",
    workspace=".",
)

print("Sending message to deserialized conversation...")
conversation.send_message("Finally, clean up by deleting both files.")
conversation.run()

print("=" * 100)
print("Conversation finished with LLM Summarizing Condenser.")
print(f"Total LLM messages collected: {len(llm_messages)}")
print("\nThe condenser automatically summarized older conversation history")
print("when the conversation exceeded the configured max_size threshold.")
print("This helps manage context length while preserving important information.")

# Report cost
cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
print(f"EXAMPLE_COST: {cost}")
```

You can run the example code as-is.

<Note>
  The model name should follow the [LiteLLM convention](https://models.litellm.ai/): `provider/model_name` (e.g., `anthropic/claude-sonnet-4-5-20250929`, `openai/gpt-4o`).
  The `LLM_API_KEY` should be the API key for your chosen provider.
</Note>

<CodeGroup>
  <CodeBlock language="bash" filename="Bring-your-own provider key" icon="terminal" wrap>
    {`export LLM_API_KEY="your-api-key"\nexport LLM_MODEL="anthropic/claude-sonnet-4-5-20250929"  # or openai/gpt-4o, etc.\ncd software-agent-sdk\nuv run python ${path_to_script_0}`}
  </CodeBlock>

  <CodeBlock language="bash" filename="OpenHands Cloud" icon="terminal" wrap>
    {`# https://app.all-hands.dev/settings/api-keys\nexport LLM_API_KEY="your-openhands-api-key"\nexport LLM_MODEL="openhands/claude-sonnet-4-5-20250929"\ncd software-agent-sdk\nuv run python ${path_to_script_0}`}
  </CodeBlock>
</CodeGroup>

<Tip>
  **ChatGPT Plus/Pro subscribers**: You can use `LLM.subscription_login()` to authenticate with your ChatGPT account and access Codex models without consuming API credits. See the [LLM Subscriptions guide](/sdk/guides/llm-subscriptions) for details.
</Tip>

## Next Steps

* **[LLM Metrics](/sdk/guides/metrics)** - Track token usage reduction and analyze cost savings


# 这是https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/context/condenser/base.py链接的内容

software-agent-sdk/openhands-sdk/openhands/sdk/context/condenser
/base.py
csmith49Calvin Smithopenhands-agent
feat(condenser): Hard context reset on unrecoverable error (#1596)
b79d72f
 · 
2 months ago
software-agent-sdk/openhands-sdk/openhands/sdk/context/condenser
/base.py

Code

Blame
184 lines (141 loc) · 7.28 KB
class RollingCondenser(PipelinableCondenserBase, ABC):
    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger

from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
)


logger = getLogger(__name__)


class CondenserBase(DiscriminatedUnionMixin, ABC):
    """Abstract condenser interface.

    Condensers take a list of `Event` objects and reduce them into a potentially smaller
    list.

    Agents can use condensers to reduce the amount of events they need to consider when
    deciding which action to take. To use a condenser, agents can call the
    `condensed_history` method on the current `State` being considered and use the
    results instead of the full history.

    If the condenser returns a `Condensation` instead of a `View`, the agent should
    return `Condensation.action` instead of producing its own action. On the next agent
    step the condenser will use that condensation event to produce a new `View`.
    """

    @abstractmethod
    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        """Condense a sequence of events into a potentially smaller list.

        New condenser strategies should override this method to implement their own
        condensation logic. Call `self.add_metadata` in the implementation to record any
        relevant per-condensation diagnostic information.

        Args:
            view: A view of the history containing all events that should be condensed.
            agent_llm: LLM instance used by the agent. Condensers use this for token
                counting purposes. Defaults to None.

        Returns:
            View | Condensation: A condensed view of the events or an event indicating
            the history has been condensed.
        """

    def handles_condensation_requests(self) -> bool:
        """Whether this condenser handles explicit condensation requests.

        If this returns True, the agent will trigger the condenser whenever a
        CondensationRequest event is added to the history. If False, the condenser will
        only be triggered when the agent's own logic decides to do so (e.g. context
        window exceeded).

        Returns:
            bool: True if the condenser handles explicit condensation requests, False
            otherwise.
        """
        return False


class PipelinableCondenserBase(CondenserBase):
    """Abstract condenser interface which may be pipelined. (Since a pipeline
    condenser should not nest another pipeline condenser)"""


class NoCondensationAvailableException(Exception):
    """Raised when a condenser is asked to provide a condensation but none is available.

    This can happen if the condenser's `should_condense` method returns True, but due to
    API constraints no condensation can be generated.

    When this exception is raised from a rolling condenser's `get_condensation` method,
    the agent will fall back to using the uncondensed view for the next agent step.
    """


class CondensationRequirement(Enum):
    """The type of condensation required by a rolling condenser."""

    HARD = "hard"
    """Indicates that a condensation is required right now, and the agent cannot proceed
    without it.
    """

    SOFT = "soft"
    """Indicates that a condensation is desired but not strictly required."""


class RollingCondenser(PipelinableCondenserBase, ABC):
    """Base class for a specialized condenser strategy that applies condensation to a
    rolling history.

    The rolling history is generated by `View.from_events`, which analyzes all events in
    the history and produces a `View` object representing what will be sent to the LLM.

    If `condensation_requirement` says so, the condenser is then responsible for
    generating a `Condensation` object from the `View` object. This will be added to the
    event history which should -- when given to `get_view` -- produce the condensed
    `View` to be passed to the LLM.
    """

    def hard_context_reset(
        self,
        view: View,  # noqa: ARG002
        agent_llm: LLM | None = None,  # noqa: ARG002
    ) -> Condensation | None:
        """Perform a hard context reset, if supported by the condenser.

        By default, rolling condensers do not support hard context resets. Override this
        method to implement hard context reset logic by returning a `Condensation`
        object.

        This method is invoked when:
        - A HARD condensation requirement is triggered (e.g., by user request)
        - But the condenser raises a NoCondensationAvailableException error
        """
        return None

    @abstractmethod
    def condensation_requirement(
        self, view: View, agent_llm: LLM | None = None
    ) -> CondensationRequirement | None:
        """Determine how a view should be condensed.

        Args:
            view: The current view of the conversation history.
            agent_llm: LLM instance used by the agent. Condensers use this for token
                counting purposes. Defaults to None.

        Returns:
            CondensationRequirement | None: The type of condensation required, or None
            if no condensation is needed.
        """

    @abstractmethod
    def get_condensation(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation:
        """Get the condensation from a view."""

    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        # If we trigger the condenser-specific condensation threshold, compute and
        # return the condensation.
        request = self.condensation_requirement(view, agent_llm=agent_llm)
        if request is not None:
            try:
                return self.get_condensation(view, agent_llm=agent_llm)

            except NoCondensationAvailableException as e:
                logger.debug(f"No condensation available: {e}")

                if request == CondensationRequirement.SOFT:
                    # For soft requests, we can just return the uncondensed view. This
                    # request will _eventually_ be handled, but it's not critical that
                    # we do so immediately.
                    return view

                elif request == CondensationRequirement.HARD:
                    # The agent has found itself in a situation where it cannot proceed
                    # without condensation, but the condenser cannot provide one. We'll
                    # try to recover from this situation by performing a hard context
                    # reset, if supported by the condenser.
                    try:
                        hard_reset_condensation = self.hard_context_reset(
                            view, agent_llm=agent_llm
                        )
                        if hard_reset_condensation is not None:
                            return hard_reset_condensation

                    # And if something goes wrong with the hard reset make sure we keep
                    # both errors in the stack
                    except Exception as hard_reset_exception:
                        raise hard_reset_exception from e

                # In all other situations re-raise the exception.
                raise e

        # Otherwise we're safe to just return the view.
        else:
            return view
