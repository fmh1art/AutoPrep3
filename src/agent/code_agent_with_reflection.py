"""
CodeAgentWithReflection — 带有逐步反思机制的 Agent。

在每个 action 执行后，使用 reviewer agent 进行反思，决定是否保留该步骤。
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.conversation import RemoteConversation
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.llm import Metrics
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace

from src.benchmarks.utils.reflection_utils import (
    ReflectionResult,
    agent_finished_with_finish_action,
    agent_sent_message,
    clone_metrics,
    collect_repo_diff,
    format_action_event,
    format_observation_event,
    get_last_agent_message,
    merge_metrics,
    parse_reflection_result,
    restore_checkpoint,
    run_conversation_until_pause_or_finish,
    status_name,
    truncate,
    write_checkpoint,
)

logger = get_logger(__name__)

FakeUserResponseFn = Callable[[RemoteConversation], str]


@dataclass
class AgentResult:
    """Agent 运行结果。"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


class StepPauseController:
    """在每个 action-observation 对完成后暂停对话，以便进行反思。"""

    def __init__(self) -> None:
        self.conversation: RemoteConversation | None = None
        self.last_action: ActionEvent | None = None
        self.last_observation: ObservationEvent | None = None
        self.pending_tool_call_id: str | None = None
        self.pause_requested = False

    def bind(self, conversation: RemoteConversation) -> None:
        self.conversation = conversation

    def before_run(self) -> None:
        self.last_action = None
        self.last_observation = None
        self.pending_tool_call_id = None
        self.pause_requested = False

    def __call__(self, event: Event) -> None:
        if isinstance(event, ActionEvent) and getattr(event, "source", None) == "agent":
            self.last_action = event
            self.pending_tool_call_id = getattr(event, "tool_call_id", None)
            return

        if not isinstance(event, ObservationEvent) or self.last_action is None:
            return

        if self.pause_requested:
            return

        observation_tool_call_id = getattr(event, "tool_call_id", None)
        if (
            self.pending_tool_call_id is not None
            and observation_tool_call_id is not None
            and observation_tool_call_id != self.pending_tool_call_id
        ):
            return

        self.last_observation = event
        self.pause_requested = True
        if self.conversation is not None:
            self.conversation.pause()


def fake_user_response(conversation: RemoteConversation, encapsulate_solution: bool = False) -> str:
    """生成假的用户响应，鼓励 agent 继续工作。"""
    encaps_str = (
        "Your final answer MUST be encapsulated within <solution> and </solution>.\n"
        "For example: The answer to the question is <solution> 42 </solution>.\n"
    ) if encapsulate_solution else ""
    msg = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "When you think you have solved the question, please use the finish tool and "
        "include your final answer in the message parameter of the finish tool.\n"
        f"{encaps_str}"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )
    user_msgs = [e for e in conversation.state.events if isinstance(e, MessageEvent) and e.source == "user"]
    if len(user_msgs) >= 2:
        return msg + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
    return msg


def reflect_on_action(
    reviewer: Agent,
    workspace,
    repo_path: str,
    action_event: ActionEvent,
    observation_event: ObservationEvent | None,
    step_index: int,
    task: str,
) -> tuple[ReflectionResult, Metrics | None]:
    """使用 reviewer agent 对单个 action 进行反思评估。"""
    logger.info("Starting reflection for step %d: tool=%s", step_index, getattr(action_event, "tool_name", None))

    review_prompt = f"""
You are a strict reviewer for an autonomous coding agent.

Decide whether the latest action should be kept.
Approve only if the action moved the task forward and did not introduce an obviously bad or redundant step.

Return JSON only in this schema:
{{
  "approved": true or false,
  "reason": "short explanation",
  "guidance": "what the agent should do next if rejected"
}}

Task goal:
{task}

Step index: {step_index}

Action event:
{format_action_event(action_event)}

Observation event:
{format_observation_event(observation_event) if observation_event is not None else 'null'}

Current workspace diff:
{collect_repo_diff(workspace, repo_path)}
""".strip()

    review_conversation = Conversation(agent=reviewer, workspace=workspace)
    review_conversation.send_message(review_prompt)
    review_conversation.run(timeout=300)
    raw_message = get_last_agent_message(list(review_conversation.state.events))
    review_metrics = review_conversation.conversation_stats.get_combined_metrics()
    logger.info("Reflection raw response for step %d: %s", step_index, truncate(raw_message, 1000))
    result = parse_reflection_result(raw_message)
    logger.info("Reflection verdict for step %d: approved=%s reason=%s", step_index, result.approved, result.reason)
    return result, review_metrics


def run_conversation_with_step_reflection(
    conversation: RemoteConversation,
    workspace,
    repo_path: str,
    reviewer: Agent,
    step_controller: StepPauseController,
    fake_user_response_fn: FakeUserResponseFn = fake_user_response,
    max_fake_responses: int = 10,
    task: str = "",
) -> Metrics | None:
    """运行带有逐步反思的对话循环。"""
    fake_response_count = 0
    run_timeout = int(os.getenv("CONVERSATION_TIMEOUT", "3600"))
    step_index = 0
    checkpoint_dir = f"/tmp/agent-checkpoint-{uuid.uuid4().hex}"
    reflection_metrics: Metrics | None = None

    write_checkpoint(workspace, repo_path, checkpoint_dir)

    while True:
        step_controller.before_run()
        logger.info("Running agent loop; next reflected step will be %d", step_index + 1)
        run_conversation_until_pause_or_finish(conversation, timeout=run_timeout)

        events = list(conversation.state.events)
        status = conversation.state.execution_status

        if agent_finished_with_finish_action(events):
            logger.info("Agent finished with FinishAction after %d reflected steps", step_index)
            break

        if step_controller.last_action is not None:
            step_index += 1
            logger.info("Captured step %d action for review: tool=%s", step_index, getattr(step_controller.last_action, "tool_name", None))
            review, review_metrics = reflect_on_action(
                reviewer=reviewer,
                workspace=workspace,
                repo_path=repo_path,
                action_event=step_controller.last_action,
                observation_event=step_controller.last_observation,
                step_index=step_index,
                task=task,
            )
            reflection_metrics = merge_metrics(reflection_metrics, review_metrics)

            if review.approved:
                logger.info("Step %d approved: %s", step_index, review.reason)
                write_checkpoint(workspace, repo_path, checkpoint_dir)
            else:
                logger.warning("Step %d rejected: %s", step_index, review.reason)
                restore_checkpoint(workspace, repo_path, checkpoint_dir)
                conversation.send_message(
                    "Your previous action was rejected by a reviewer.\n"
                    f"Reason: {review.reason}\n"
                    f"Guidance: {review.guidance or 'Choose a different action.'}\n"
                    "Do not repeat the rejected action. Continue from the current workspace state."
                )
            continue

        if status_name(status) == "finished":
            if not agent_sent_message(events):
                logger.warning("Conversation finished without FinishAction or agent message")
                break

            if fake_response_count >= max_fake_responses:
                logger.warning("Reached maximum fake responses (%d), stopping conversation", max_fake_responses)
                break

            fake_response = fake_user_response_fn(conversation)
            if fake_response == "/exit":
                logger.info("Fake user response function returned /exit, stopping")
                break

            logger.info("Sending fake user response #%d: %s...", fake_response_count + 1, fake_response[:50])
            conversation.send_message(fake_response)
            fake_response_count += 1
            continue

        if status_name(status) == "paused":
            logger.warning("Conversation paused without a completed action to review")
            break

        logger.info("Conversation ended with status: %s after %d fake responses", getattr(status, "value", status), fake_response_count)
        break

    logger.info("Conversation completed. Total fake responses sent: %d, reflected steps: %d", fake_response_count, step_index)
    return reflection_metrics


class CodeAgentWithReflection:
    """
    带有逐步反思机制的 CodeAgent。

    在每个 action 执行后，使用 reviewer agent 进行反思，决定是否保留该步骤。
    接口与 CodeAgent 保持一致。

    Args:
        llm:                   主 Agent 使用的 LLM 实例
        reviewer_llm:          Reviewer Agent 使用的 LLM 实例（可选，默认使用主 LLM）
        tools:                 工具列表
        system_prompt_kwargs:  传递给 Agent 的 system_prompt 参数
    """

    def __init__(
        self,
        llm: LLM,
        reviewer_llm: LLM | None = None,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
    ):
        self.llm = llm
        self.reviewer_llm = reviewer_llm or llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
        self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
    ) -> AgentResult:
        """
        运行带反思的 Agent 完成一次对话。

        Args:
            instruction:  发送给 Agent 的任务指令
            workspace:    Docker 工作空间
            callbacks:    事件回调列表（用于保存轨迹等）
            repo_path:    仓库路径（用于 checkpoint 和 diff）

        Returns:
            AgentResult，包含 metrics 和 conversation 引用
        """
        agent = Agent(llm=self.llm, tools=self.tools, system_prompt_kwargs=self.system_prompt_kwargs)
        reviewer = Agent(llm=self.reviewer_llm, tools=[])

        step_controller = StepPauseController()
        conversation = Conversation(agent=agent, workspace=workspace, callbacks=[step_controller] + (callbacks or []))
        step_controller.bind(conversation)

        conversation.send_message(instruction)
        reflection_metrics = run_conversation_with_step_reflection(
            conversation=conversation,
            workspace=workspace,
            repo_path=repo_path,
            reviewer=reviewer,
            step_controller=step_controller,
            task=instruction,
        )

        main_metrics = conversation.conversation_stats.get_combined_metrics()
        total_metrics = merge_metrics(main_metrics, reflection_metrics)
        metrics_data = self._extract_metrics(total_metrics)

        return AgentResult(metrics=metrics_data, conversation=conversation)

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """从 conversation metrics 中提取数据。"""
        if metrics is None:
            return {}

        metrics_data: dict[str, Any] = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage is not None:
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
            metrics_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0),
                "total_tokens": prompt_tokens + completion_tokens,
            }
        metrics_data["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return metrics_data

