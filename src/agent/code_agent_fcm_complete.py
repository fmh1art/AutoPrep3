"""
完整的FCM实现 - 包括action验证和修正

流程：
1. A_code生成T_0, A_0，执行得到O_0
2. A_context验证A_0，必要时修正为A'_0
3. Summarize T_0 → T'_0, O_0 → O''_0
4. 用T'_0, A'_0, O''_0替换trajectory
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import litellm

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.llm.message import TextContent
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools

logger = get_logger(__name__)

_METRIC_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)


def _empty_metrics() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "accumulated_cost": 0.0,
    }


def _merge_metrics(*metrics_list: dict[str, Any] | None) -> dict[str, Any]:
    merged = _empty_metrics()
    for metrics in metrics_list:
        if not metrics:
            continue
        for key in _METRIC_KEYS:
            merged[key] += int(metrics.get(key, 0) or 0)
        merged["accumulated_cost"] += float(metrics.get("accumulated_cost", 0.0) or 0.0)
    return merged


def _extract_response_metrics(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    metrics = _empty_metrics()
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        metrics.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": int(getattr(usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(usage, "cache_write_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) or (prompt_tokens + completion_tokens),
            }
        )

    hidden_params = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        metrics["accumulated_cost"] = float(hidden_params.get("response_cost", 0.0) or 0.0)

    return metrics


def _to_plain_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_to_plain_text(item) for item in value)

    text = getattr(value, "text", None)
    if text is not None:
        return str(text)

    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _to_plain_text(content)

    to_llm_content = getattr(value, "to_llm_content", None)
    if to_llm_content is not None:
        try:
            return "".join(_to_plain_text(item) for item in to_llm_content)
        except TypeError:
            return _to_plain_text(to_llm_content)

    return str(value)


def _unwrap_secret(value: Any) -> Any:
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return value


def _build_observation_event(
    original_event: ObservationEvent,
    summarized_observation: str,
) -> ObservationEvent | None:
    original_observation = original_event.observation

    try:
        if hasattr(original_observation, "model_dump"):
            payload = original_observation.model_dump()
            payload["content"] = [TextContent(text=summarized_observation)]
            new_observation = type(original_observation)(**payload)
        else:
            new_observation = type(original_observation)(content=[TextContent(text=summarized_observation)])
    except Exception as exc:
        logger.warning(f"[FCM] Failed to rebuild observation payload: {exc}")
        return None

    new_event = ObservationEvent(
        id=original_event.id,
        observation=new_observation,
        source=original_event.source,
        timestamp=original_event.timestamp,
        tool_name=original_event.tool_name,
        tool_call_id=original_event.tool_call_id,
        action_id=original_event.action_id,
    )
    return new_event


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


class ActionVerifier:
    """验证和修正action的A_context agent"""

    def __init__(self, context_llm: LLM, workspace: DockerWorkspace, tools: list):
        self.context_llm = context_llm
        self.workspace = workspace
        self.tools = tools

    def verify_and_fix_action(
        self,
        thinking: str,
        action_event: ActionEvent,
        observation: str,
        history: list
    ) -> tuple[ActionEvent, bool, dict[str, Any]]:
        """验证action是否正确，必要时修正"""
        thinking = _to_plain_text(thinking)
        observation = _to_plain_text(observation)
        action_str = self._format_action(action_event)

        verification_prompt = f"""Review this agent action:

Thinking: {thinking[:500]}
Action: {action_str}
Observation: {observation[:500]}

Is this action correct? Respond:
CORRECT: yes/no
REASON: <brief explanation>
"""

        try:
            model_name = self.context_llm.model

            response = litellm.completion(
                model=model_name,
                api_key=_unwrap_secret(self.context_llm.api_key),
                base_url=_unwrap_secret(self.context_llm.base_url),
                messages=[{"role": "user", "content": verification_prompt}],
                temperature=0.3,
                max_tokens=300
            )

            result = response.choices[0].message.content
            metrics = _extract_response_metrics(response)
            logger.info(f"[FCM] Action verification: {result[:200]}")

            if "CORRECT: no" in result.lower():
                logger.info(f"[FCM] Action needs correction")
                return action_event, True, metrics

            return action_event, False, metrics

        except Exception as e:
            logger.error(f"[FCM] Action verification failed: {e}")
            return action_event, False, _empty_metrics()

    def _format_action(self, action_event: ActionEvent) -> str:
        """格式化action为可读字符串"""
        action = action_event.action
        if hasattr(action, 'name'):
            return f"{action.name}({getattr(action, 'arguments', '')})"
        return str(action)


def summarize_thinking(
    text: Any,
    context_llm: LLM,
    threshold: int = 80,
) -> tuple[str, dict[str, Any]]:
    """Summarize thinking if exceeds threshold"""
    text = _to_plain_text(text)
    word_count = len(text.split())
    if word_count <= threshold:
        return text, _empty_metrics()

    try:
        logger.info(f"[FCM] Summarizing thinking ({word_count} words)")
        model_name = context_llm.model

        prompt = f"Summarize in 2 sentences:\n{text[:2000]}"
        response = litellm.completion(
            model=model_name,
            api_key=_unwrap_secret(context_llm.api_key),
            base_url=_unwrap_secret(context_llm.base_url),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content, _extract_response_metrics(response)
    except Exception as e:
        logger.error(f"[FCM] Thinking summarization failed: {e}")
        return text[:200], _empty_metrics()


def summarize_observation(
    text: Any,
    context_llm: LLM,
    threshold: int = 100,
) -> tuple[str, dict[str, Any]]:
    """Summarize observation if exceeds threshold"""
    text = _to_plain_text(text)
    word_count = len(text.split())
    if word_count <= threshold:
        return text, _empty_metrics()

    try:
        logger.info(f"[FCM] Summarizing observation ({word_count} words)")
        model_name = context_llm.model

        prompt = f"Summarize in 3 sentences:\n{text[:2000]}"
        response = litellm.completion(
            model=model_name,
            api_key=_unwrap_secret(context_llm.api_key),
            base_url=_unwrap_secret(context_llm.base_url),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content, _extract_response_metrics(response)
    except Exception as e:
        logger.error(f"[FCM] Observation summarization failed: {e}")
        return text[:300], _empty_metrics()


class CodeAgentWithFCM:
    """完整的FCM Agent - 包括action验证"""

    def __init__(self, code_llm: LLM, context_llm: LLM, tools: list | None = None):
        self.code_llm = code_llm
        self.context_llm = context_llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
        self.thinking_threshold = 80
        self.obs_threshold = 100

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        log_dir: str | None = None,
    ) -> AgentResult:
        """运行完整FCM agent"""

        logger.info("[FCM] Starting agent with complete FCM (action verification)")
        fcm_metrics = _empty_metrics()

        # 创建action verifier
        verifier = ActionVerifier(self.context_llm, workspace, self.tools)

        # 创建标准agent
        agent = Agent(
            llm=self.code_llm,
            tools=self.tools,
            system_prompt_kwargs={"cli_mode": True}
        )

        # 创建FCM callback
        def fcm_callback(event):
            """在每个ObservationEvent后应用FCM"""
            nonlocal fcm_metrics
            try:
                if not isinstance(event, ObservationEvent):
                    return

                events = list(conversation.state.events)
                if len(events) < 2:
                    return

                # 找最后的action和observation
                last_action_event = None
                last_obs_event = None
                last_action_idx = None
                last_obs_idx = None

                for i in range(len(events) - 1, -1, -1):
                    if last_obs_event is None and isinstance(events[i], ObservationEvent):
                        last_obs_event = events[i]
                        last_obs_idx = i
                    if last_action_event is None and isinstance(events[i], ActionEvent):
                        last_action_event = events[i]
                        last_action_idx = i
                    if last_action_event and last_obs_event:
                        break

                if not last_action_event or not last_obs_event:
                    return

                # 提取T_0, A_0, O_0
                thinking = _to_plain_text(getattr(last_action_event, 'thought', ''))
                observation = _to_plain_text(getattr(last_obs_event, 'observation', ''))

                # A_context验证action
                history = events[:last_action_idx]
                corrected_action, _was_modified, verification_metrics = verifier.verify_and_fix_action(
                    thinking, last_action_event, observation, history
                )
                fcm_metrics = _merge_metrics(fcm_metrics, verification_metrics)

                # Summarize
                summarized_thinking, thinking_metrics = summarize_thinking(
                    thinking, self.context_llm, self.thinking_threshold
                )
                fcm_metrics = _merge_metrics(fcm_metrics, thinking_metrics)

                summarized_obs, observation_metrics = summarize_observation(
                    observation, self.context_llm, self.obs_threshold
                )
                fcm_metrics = _merge_metrics(fcm_metrics, observation_metrics)

                # 替换events
                modified = False
                if summarized_thinking != thinking:
                    new_action_event = ActionEvent(
                        action=corrected_action.action,
                        thought=summarized_thinking,
                        source=last_action_event.source,
                        timestamp=last_action_event.timestamp
                    )
                    for attr in ['tool_call_id', 'action_id']:
                        if hasattr(last_action_event, attr):
                            setattr(new_action_event, attr, getattr(last_action_event, attr))
                    events[last_action_idx] = new_action_event
                    modified = True

                if summarized_obs != observation:
                    new_obs_event = _build_observation_event(last_obs_event, summarized_obs)
                    if new_obs_event is not None:
                        events[last_obs_idx] = new_obs_event
                        modified = True

                if modified:
                    conversation.state.events = events
            except Exception:
                logger.exception("[FCM] Callback failed; preserving original conversation events")

        # 添加FCM callback
        all_callbacks = (callbacks or []) + [fcm_callback]

        # 创建conversation
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=all_callbacks
        )

        # 发送消息
        conversation.send_message(instruction)

        # 运行conversation
        from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
        run_conversation_with_fake_user_response(
            conversation=conversation,
            max_fake_responses=50
        )

        # 收集metrics
        code_agent_metrics = self._extract_metrics(conversation.conversation_stats.get_combined_metrics())
        total_metrics = _merge_metrics(code_agent_metrics, fcm_metrics)
        metrics_data = {
            **total_metrics,
            "code_agent_metrics": code_agent_metrics,
            "fcm_metrics": fcm_metrics,
        }

        logger.info("[FCM] Agent completed")
        return AgentResult(metrics=metrics_data, conversation=conversation)

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """提取metrics"""
        if not metrics:
            return {}

        result = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage:
            p = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            c = int(getattr(token_usage, "completion_tokens", 0) or 0)
            result = {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": p + c,
                "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0),
            }
        result["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return result






