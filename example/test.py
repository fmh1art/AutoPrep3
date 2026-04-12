import json
import os
import platform
import shlex
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import yaml
from openhands.sdk import Agent, Conversation, get_logger, LLM
from openhands.sdk.conversation import RemoteConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.llm import Metrics, content_to_str
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)

FakeUserResponseFn = Callable[[RemoteConversation], str]


@dataclass
class ReflectionResult:
    approved: bool
    reason: str
    guidance: str = ""


class StepPauseController:
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

def fake_user_response(
    conversation: RemoteConversation,
    encapsulate_solution: bool = False,
) -> str:
    encaps_str = (
        (
            "Your final answer MUST be encapsulated within <solution> and </solution>.\n"
            "For example: The answer to the question is <solution> 42 </solution>.\n"
        )
        if encapsulate_solution
        else ""
    )
    msg = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "When you think you have solved the question, please use the finish tool and "
        "include your final answer in the message parameter of the finish tool.\n"
        f"{encaps_str}"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )

    user_msgs = [
        e for e in conversation.state.events
        if isinstance(e, MessageEvent) and e.source == "user"
    ]
    if len(user_msgs) >= 2:
        return msg + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
    return msg


def _agent_finished_with_finish_action(events: list[Event]) -> bool:
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            return event.action is not None and isinstance(event.action, FinishAction)
    return False


def _agent_sent_message(events: list[Event]) -> bool:
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            return False
    return False


def _should_stop(
    conversation: RemoteConversation,
    fake_response_count: int,
    max_fake_responses: int,
) -> bool:
    status = conversation.state.execution_status
    if status != ConversationExecutionStatus.FINISHED:
        logger.info(
            "Conversation ended with status: %s after %d fake responses",
            status.value,
            fake_response_count,
        )
        return True

    events = list(conversation.state.events)

    if _agent_finished_with_finish_action(events):
        logger.info(
            "Agent finished with FinishAction after %d fake responses",
            fake_response_count,
        )
        return True

    if not _agent_sent_message(events):
        logger.warning("Conversation finished without FinishAction or agent message")
        return True

    if fake_response_count >= max_fake_responses:
        logger.warning(
            "Reached maximum fake responses (%d), stopping conversation",
            max_fake_responses,
        )
        return True

    return False


def _status_name(status: object) -> str:
    return str(getattr(status, "value", status)).lower()


def _get_last_agent_message(events: list[Event]) -> str:
    for event in reversed(events):
        if not isinstance(event, MessageEvent) or getattr(event, "source", None) != "agent":
            continue

        llm_message = getattr(event, "llm_message", None)
        if llm_message is not None:
            try:
                return "".join(content_to_str(llm_message.content)).strip()
            except Exception:
                logger.debug("Failed to decode llm_message content", exc_info=True)

        message = getattr(event, "message", None)
        if message is not None:
            return str(message).strip()

        text = getattr(event, "content", None)
        if text is not None:
            return str(text).strip()

        return str(event).strip()

    return ""


def _quote(value: str) -> str:
    return shlex.quote(value)


def _run_workspace_command(
    workspace: RemoteWorkspace,
    command: str,
    timeout: float = 60.0,
    allow_failure: bool = False,
) -> str:
    result = workspace.execute_command(command, timeout=timeout)
    if result.exit_code != 0 and not allow_failure:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return (result.stdout or "") + ((result.stderr or "") if allow_failure else "")


def _run_conversation_until_pause_or_finish(
    conversation: RemoteConversation,
    timeout: int,
    join_timeout: float = 30.0,
) -> None:
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            conversation.run(timeout=timeout)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=join_timeout)

    if thread.is_alive():
        logger.warning(
            "conversation.run() still active after %.1fs; waiting for pause/finish",
            join_timeout,
        )
        thread.join()

    if errors:
        raise RuntimeError("conversation.run() failed") from errors[0]


def _write_checkpoint(
    workspace: RemoteWorkspace,
    repo_path: str,
    checkpoint_dir: str,
) -> None:
    repo = _quote(repo_path)
    checkpoint = _quote(checkpoint_dir)
    archive = _quote(f"{checkpoint_dir}/repo.tar")
    command = (
        f"rm -rf {checkpoint} && mkdir -p {checkpoint} && "
        f"tar --exclude='.git' -C {repo} -cf {archive} ."
    )
    _run_workspace_command(workspace, command, timeout=120.0)


def _restore_checkpoint(
    workspace: RemoteWorkspace,
    repo_path: str,
    checkpoint_dir: str,
) -> None:
    repo = _quote(repo_path)
    archive = _quote(f"{checkpoint_dir}/repo.tar")
    command = (
        f"find {repo} -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {{}} + && "
        f"tar -C {repo} -xf {archive}"
    )
    _run_workspace_command(workspace, command, timeout=120.0)


def _truncate(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            logger.debug("Failed to model_dump value for JSON conversion", exc_info=True)

    if hasattr(value, "content"):
        try:
            content = getattr(value, "content")
            return "".join(content_to_str(content)).strip()
        except Exception:
            logger.debug("Failed to stringify content field", exc_info=True)

    try:
        return str(value)
    except Exception:
        return repr(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _clone_metrics(metrics: Metrics | None) -> Metrics | None:
    if metrics is None:
        return None
    return metrics.model_copy(deep=True)


def _merge_metrics(*items: Metrics | None) -> Metrics | None:
    merged: Metrics | None = None
    for item in items:
        if item is None:
            continue
        if merged is None:
            merged = _clone_metrics(item)
            continue
        merged.merge(item)
    return merged


def _print_metrics_summary(label: str, metrics: object) -> None:
    if metrics is None:
        print(f"[METRICS] {label}: no metrics available")
        return

    accumulated_cost = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
    token_usage = getattr(metrics, "accumulated_token_usage", None)
    if token_usage is None:
        print(
            f"[METRICS] {label}: cost=${_format_float(accumulated_cost)}, no token usage available"
        )
        return

    prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
    reasoning_tokens = int(getattr(token_usage, "reasoning_tokens", 0) or 0)
    cache_read_tokens = int(getattr(token_usage, "cache_read_tokens", 0) or 0)
    cache_write_tokens = int(getattr(token_usage, "cache_write_tokens", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens

    print(
        "[METRICS] "
        f"{label}: "
        f"prompt={prompt_tokens}, "
        f"completion={completion_tokens}, "
        f"total={total_tokens}, "
        f"reasoning={reasoning_tokens}, "
        f"cache_read={cache_read_tokens}, "
        f"cache_write={cache_write_tokens}, "
        f"cost=${_format_float(accumulated_cost)}"
    )


def _format_action_event(event: ActionEvent) -> str:
    action = getattr(event, "action", None)
    payload = {
        "tool_name": getattr(event, "tool_name", None),
        "tool_call_id": getattr(event, "tool_call_id", None),
        "thought": getattr(action, "thought", None),
        "command": getattr(action, "command", None),
        "path": getattr(action, "path", None),
    }
    return json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)


def _format_observation_event(event: ObservationEvent) -> str:
    observation = getattr(event, "observation", None)
    payload = {
        "tool_call_id": getattr(event, "tool_call_id", None),
        "content": getattr(observation, "content", None),
        "stdout": getattr(observation, "stdout", None),
        "stderr": getattr(observation, "stderr", None),
        "command": getattr(observation, "command", None),
        "path": getattr(observation, "path", None),
        "exit_code": getattr(observation, "exit_code", None),
        "is_error": getattr(observation, "is_error", None),
    }
    return json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)


def _collect_repo_diff(workspace: RemoteWorkspace, repo_path: str) -> str:
    repo = _quote(repo_path)
    command = (
        f"cd {repo} && "
        "git status --short && "
        "printf '\n--- DIFF STAT ---\n' && git diff --no-color --stat && "
        "printf '\n--- DIFF PATCH ---\n' && git diff --no-color --unified=0"
    )
    output = _run_workspace_command(workspace, command, timeout=60.0, allow_failure=True)
    return _truncate(output.strip() or "No working tree diff.")


def _parse_reflection_result(raw_message: str) -> ReflectionResult:
    try:
        start = raw_message.find("{")
        end = raw_message.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in reviewer output")
        payload = json.loads(raw_message[start : end + 1])
        return ReflectionResult(
            approved=bool(payload.get("approved")),
            reason=str(payload.get("reason") or "No reason provided."),
            guidance=str(payload.get("guidance") or ""),
        )
    except Exception as exc:
        logger.warning("Failed to parse reviewer output: %s", exc)
        return ReflectionResult(
            approved=False,
            reason="Reviewer output was not valid JSON.",
            guidance="Return strict JSON with approved/reason/guidance.",
        )


def reflect_on_action(
    reviewer: Agent,
    workspace: RemoteWorkspace,
    repo_path: str,
    action_event: ActionEvent,
    observation_event: ObservationEvent | None,
    step_index: int,
    task: str,
) -> tuple[ReflectionResult, Metrics | None]:
    logger.info(
        "Starting reflection for step %d: tool=%s tool_call_id=%s",
        step_index,
        getattr(action_event, "tool_name", None),
        getattr(action_event, "tool_call_id", None),
    )

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
{_format_action_event(action_event)}

Observation event:
{_format_observation_event(observation_event) if observation_event is not None else 'null'}

Current workspace diff:
{_collect_repo_diff(workspace, repo_path)}
""".strip()

    review_conversation = Conversation(agent=reviewer, workspace=workspace)
    review_conversation.send_message(review_prompt)
    review_conversation.run(timeout=300)
    raw_message = _get_last_agent_message(list(review_conversation.state.events))
    review_metrics = review_conversation.conversation_stats.get_combined_metrics()
    _print_metrics_summary(
        f"reflection-step-{step_index}",
        review_metrics,
    )
    logger.info("Reflection raw response for step %d: %s", step_index, _truncate(raw_message, 1000))
    result = _parse_reflection_result(raw_message)
    logger.info(
        "Reflection verdict for step %d: approved=%s reason=%s guidance=%s",
        step_index,
        result.approved,
        result.reason,
        result.guidance or "<empty>",
    )
    return result, review_metrics


def run_conversation_with_step_reflection(
    conversation: RemoteConversation,
    workspace: RemoteWorkspace,
    repo_path: str,
    reviewer: Agent,
    step_controller: StepPauseController,
    fake_user_response_fn: FakeUserResponseFn = fake_user_response,
    max_fake_responses: int = 10,
    task: str = "",
) -> Metrics | None:
    fake_response_count = 0
    run_timeout = 3600
    step_index = 0
    checkpoint_dir = f"/tmp/agent-checkpoint-{uuid.uuid4().hex}"
    reflection_metrics: Metrics | None = None

    _write_checkpoint(workspace, repo_path, checkpoint_dir)

    while True:
        step_controller.before_run()
        logger.info("Running agent loop; next reflected step will be %d", step_index + 1)
        _run_conversation_until_pause_or_finish(conversation, timeout=run_timeout)

        events = list(conversation.state.events)
        status = conversation.state.execution_status

        if _agent_finished_with_finish_action(events):
            logger.info("Agent finished with FinishAction after %d reflected steps", step_index)
            break

        if step_controller.last_action is not None:
            step_index += 1
            logger.info(
                "Captured step %d action for review: tool=%s",
                step_index,
                getattr(step_controller.last_action, "tool_name", None),
            )
            review, review_metrics = reflect_on_action(
                reviewer=reviewer,
                workspace=workspace,
                repo_path=repo_path,
                action_event=step_controller.last_action,
                observation_event=step_controller.last_observation,
                step_index=step_index,
                task=task,
            )
            reflection_metrics = _merge_metrics(reflection_metrics, review_metrics)

            if review.approved:
                logger.info("Step %d approved: %s", step_index, review.reason)
                _write_checkpoint(workspace, repo_path, checkpoint_dir)
            else:
                logger.warning("Step %d rejected: %s", step_index, review.reason)
                _restore_checkpoint(workspace, repo_path, checkpoint_dir)
                conversation.send_message(
                    "Your previous action was rejected by a reviewer.\n"
                    f"Reason: {review.reason}\n"
                    f"Guidance: {review.guidance or 'Choose a different action.'}\n"
                    "Do not repeat the rejected action. Continue from the current workspace state."
                )
            continue

        if _status_name(status) == "finished":
            if not _agent_sent_message(events):
                logger.warning("Conversation finished without FinishAction or agent message")
                break

            if fake_response_count >= max_fake_responses:
                logger.warning(
                    "Reached maximum fake responses (%d), stopping conversation",
                    max_fake_responses,
                )
                break

            fake_response = fake_user_response_fn(conversation)
            if fake_response == "/exit":
                logger.info("Fake user response function returned /exit, stopping")
                break

            logger.info(
                "Sending fake user response #%d: %s...",
                fake_response_count + 1,
                fake_response[:50],
            )
            conversation.send_message(fake_response)
            fake_response_count += 1
            continue

        if _status_name(status) == "paused":
            logger.warning("Conversation paused without a completed action to review")
            break

        logger.info(
            "Conversation ended with status: %s after %d fake responses",
            getattr(status, "value", status),
            fake_response_count,
        )
        break

    logger.info(
        "Conversation completed. Total fake responses sent: %d, reflected steps: %d",
        fake_response_count,
        step_index,
    )
    return reflection_metrics


def is_acp_agent(agent_type: str) -> bool:
    return agent_type in ("claude", "codex")

@contextmanager
def workspace_keepalive(
    agent_type: str, workspace: RemoteWorkspace, interval: int = 60
):
    """Keep the runtime workspace alive during ACP agent execution."""
    if not is_acp_agent(agent_type):
        yield
        return

    stop = threading.Event()

    def _ping() -> None:
        while True:
            try:
                workspace.execute_command("true")
                logger.debug("Workspace keep-alive ping sent")
            except Exception:
                logger.debug("Workspace keep-alive ping failed", exc_info=True)
            if stop.wait(interval):
                break

    t = threading.Thread(target=_ping, daemon=True)
    t.start()
    logger.info("Started workspace keep-alive (interval=%ds)", interval)
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=5)
        logger.info("Stopped workspace keep-alive")


def run_conversation_with_fake_user_response(
    conversation: RemoteConversation,
    fake_user_response_fn: FakeUserResponseFn = fake_user_response,
    max_fake_responses: int = 10,
) -> None:
    fake_response_count = 0

    while True:
        conversation.run()

        if _should_stop(conversation, fake_response_count, max_fake_responses):
            break

        fake_response = fake_user_response_fn(conversation)
        if fake_response == "/exit":
            logger.info("Fake user response function returned /exit, stopping")
            break

        logger.info(
            "Sending fake user response #%d: %s...",
            fake_response_count + 1,
            fake_response[:50],
        )
        conversation.send_message(fake_response)
        fake_response_count += 1

    logger.info(
        "Conversation completed. Total fake responses sent: %d", fake_response_count
    )


def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


tools = get_default_tools(enable_browser=False)
cfg = yaml.safe_load(open("_config/doubao.yaml"))

llm = LLM(
    model=f"openai/{cfg['llm_name']}",
    api_key=cfg['key'],
    base_url=cfg['openai_base_url'],
)

agent = Agent(
    llm=llm,
    tools=tools,
    system_prompt_kwargs={"cli_mode": True},
)

reviewer = Agent(
    llm=llm,
    tools=[],
)

instruction = "打印项目文件目录"
agent_type = 'openai'

HTTP_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
REPO_URL = "https://github.com/fmh1art/BatchER.git"
REPO_DIR_NAME = REPO_URL.rsplit("/", 1)[-1].removesuffix(".git")

# 排除不需要走代理的域名：
#   - localhost/127.0.0.1：Docker 健康检查直连
#   - bytedance.net：内部 LLM API（ark-cn-beijing.bytedance.net），代理会返回 403
NO_PROXY_LIST = "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# 将代理写入宿主机环境变量，再通过 forward_env 注入容器
os.environ["http_proxy"] = HTTP_PROXY
os.environ["https_proxy"] = HTTP_PROXY
os.environ["HTTP_PROXY"] = HTTP_PROXY
os.environ["HTTPS_PROXY"] = HTTP_PROXY
os.environ["no_proxy"] = NO_PROXY_LIST
os.environ["NO_PROXY"] = NO_PROXY_LIST

with DockerWorkspace(
    server_image="ghcr.io/openhands/agent-server:latest-python",
    platform=detect_platform(),
    forward_env=["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"],
) as workspace:
    # 容器启动后，设置代理并 clone 项目
    logger.info("Setting proxy and cloning repo into workspace...")
    init_script = (
        f"export http_proxy={HTTP_PROXY} && "
        f"export https_proxy={HTTP_PROXY} && "
        f"export HTTP_PROXY={HTTP_PROXY} && "
        f"export HTTPS_PROXY={HTTP_PROXY} && "
        f"export no_proxy={NO_PROXY_LIST} && "
        f"export NO_PROXY={NO_PROXY_LIST} && "
        f"cd /workspace && "
        f"git clone {REPO_URL}"
    )
    result = workspace.execute_command(init_script, timeout=120.0)
    if result.exit_code != 0:
        logger.warning("Init script failed (exit %d): %s", result.exit_code, result.stderr)
    else:
        logger.info("Repo cloned successfully: %s", result.stdout)

    step_controller = StepPauseController()
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[step_controller],
    )
    step_controller.bind(conversation)
    try:
        conversation.send_message(instruction)
        with workspace_keepalive(agent_type, workspace):
            reflection_metrics = run_conversation_with_step_reflection(
                conversation=conversation,
                workspace=workspace,
                repo_path=f"/workspace/{REPO_DIR_NAME}",
                reviewer=reviewer,
                step_controller=step_controller,
                task=instruction,
            )
        main_metrics = conversation.conversation_stats.get_combined_metrics()
        total_metrics = _merge_metrics(main_metrics, reflection_metrics)
        _print_metrics_summary(
            "main-conversation",
            main_metrics,
        )
        _print_metrics_summary("reflection-total", reflection_metrics)
        _print_metrics_summary("all-conversations-total", total_metrics)
    finally:
        conversation.close()

print("All done!")
