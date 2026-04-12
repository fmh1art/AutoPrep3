"""
FCM Condenser - 使用OpenHands的Condenser机制实现真正的token节省

核心思路：
1. 继承CondenserBase
2. 在condense方法中修改events
3. 只修改最后一步，保持KV Cache有效
"""

from pydantic import Field
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM
from openhands.sdk import get_logger

logger = get_logger(__name__)


class FCMCondenser(CondenserBase):
    """FCM Condenser - 压缩thinking和observation"""

    context_llm: LLM
    thinking_threshold: int = Field(default=80)
    obs_threshold: int = Field(default=100)

    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        """压缩events中的长内容 - 只修改最后一步"""

        events = list(view.events)
        if len(events) < 2:
            return view

        # 只处理最后的action-observation对
        last_action_idx = None
        last_obs_idx = None

        # 从后往前找最后的action和observation
        for i in range(len(events) - 1, -1, -1):
            if last_obs_idx is None and isinstance(events[i], ObservationEvent):
                last_obs_idx = i
            if last_action_idx is None and isinstance(events[i], ActionEvent):
                last_action_idx = i
            if last_action_idx is not None and last_obs_idx is not None:
                break

        # 修改最后的action（thinking）
        if last_action_idx is not None:
            event = events[last_action_idx]
            thought = getattr(event, 'thought', None)
            if thought:
                thought_text = self._extract_text(thought)
                word_count = len(thought_text.split())

                if word_count > self.thinking_threshold:
                    logger.info(f"[FCM] Summarizing last thinking ({word_count} words)")
                    summarized = self._summarize_thinking(thought_text)
                    events[last_action_idx] = self._replace_thought(event, summarized)

        # 修改最后的observation
        if last_obs_idx is not None:
            event = events[last_obs_idx]
            obs = getattr(event, 'observation', None)
            if obs:
                obs_text = str(obs)
                word_count = len(obs_text.split())

                if word_count > self.obs_threshold:
                    logger.info(f"[FCM] Summarizing last observation ({word_count} words)")
                    summarized = self._summarize_observation(obs_text)
                    events[last_obs_idx] = self._replace_observation(event, summarized)

        # 返回新的View
        return View(events=events)


    def _extract_text(self, thought) -> str:
        """提取thought文本"""
        if isinstance(thought, str):
            return thought
        if isinstance(thought, list):
            return "".join(str(t) for t in thought)
        return str(thought)

    def _summarize_thinking(self, text: str) -> str:
        """总结thinking"""
        try:
            prompt = f"Summarize in 2 sentences:\n{text[:2000]}"
            response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Thinking summarization failed: {e}")
            return text[:200]

    def _summarize_observation(self, text: str) -> str:
        """总结observation"""
        try:
            prompt = f"Summarize in 3 sentences:\n{text[:2000]}"
            response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Observation summarization failed: {e}")
            return text[:300]

    def _replace_thought(self, event: ActionEvent, new_thought: str) -> ActionEvent:
        """创建新的ActionEvent with替换的thought"""
        # 复制event的属性
        new_event = ActionEvent(
            action=event.action,
            thought=new_thought,
            source=event.source,
            timestamp=event.timestamp
        )
        # 复制其他属性
        for attr in ['tool_call_id', 'action_id']:
            if hasattr(event, attr):
                setattr(new_event, attr, getattr(event, attr))
        return new_event

    def _replace_observation(self, event: ObservationEvent, new_obs: str) -> ObservationEvent:
        """创建新的ObservationEvent with替换的observation"""
        from openhands.sdk.tool import Observation

        # 创建新的observation对象
        new_observation = Observation(content=new_obs)

        new_event = ObservationEvent(
            observation=new_observation,
            source=event.source,
            timestamp=event.timestamp
        )
        # 复制其他属性
        for attr in ['tool_call_id', 'action_id']:
            if hasattr(event, attr):
                setattr(new_event, attr, getattr(event, attr))
        return new_event

    def handles_condensation_requests(self) -> bool:
        """是否处理condensation requests"""
        return False


