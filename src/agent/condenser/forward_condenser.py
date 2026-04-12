"""
Forward Context Management Condenser

Unlike backward condensers that modify historical steps (breaking KV cache),
this condenser only optimizes the CURRENT step (Ti, Ai, Oi) -> (T'i, A'i, O'i)
while keeping all previous steps 0~i-1 unchanged.
"""

from typing import Any
import litellm
from openhands.sdk import LLM, get_logger
from openhands.sdk.context.condenser.base import RollingCondenser, CondensationRequirement
from openhands.sdk.context.view import View
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event import ActionEvent, ObservationEvent

logger = get_logger(__name__)


def _unwrap_secret(value: Any) -> Any:
    """Unwrap SecretStr to plain string"""
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return value


def _to_plain_text(value: Any) -> str:
    """Convert various content types to plain text"""
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

    return str(value)


class ForwardContextCondenser(RollingCondenser):
    """
    Forward Context Management Condenser

    Optimizes only the current step while preserving KV cache:
    - Keeps steps 0~i-1 unchanged
    - Optimizes step i: (Ti, Ai, Oi) -> (T'i, A'i, O'i)
    """

    def __init__(
        self,
        context_llm: LLM,
        max_size: int = 20,
        thinking_threshold: int = 80,
        observation_threshold: int = 100,
    ):
        self.context_llm = context_llm
        self.max_size = max_size
        self.thinking_threshold = thinking_threshold
        self.observation_threshold = observation_threshold

    def condensation_requirement(
        self, view: View, agent_llm: LLM | None = None
    ) -> CondensationRequirement | None:
        """Check if condensation is needed"""
        if len(view.messages) > self.max_size:
            return CondensationRequirement.SOFT
        return None

    def get_condensation(self, view: View, agent_llm: LLM | None = None) -> Condensation:
        """
        Generate condensation for the CURRENT step only.

        Forward approach: optimize (Ti, Ai, Oi) while keeping 0~i-1 unchanged
        """
        logger.info("[FCM Condenser] Applying forward context management")

        # Find the last action-observation pair
        events = view.events
        last_action_idx = None
        last_obs_idx = None

        for i in range(len(events) - 1, -1, -1):
            if last_obs_idx is None and isinstance(events[i], ObservationEvent):
                last_obs_idx = i
            if last_action_idx is None and isinstance(events[i], ActionEvent):
                last_action_idx = i
            if last_action_idx is not None and last_obs_idx is not None:
                break

        if last_action_idx is None or last_obs_idx is None:
            # No action-observation pair to condense
            return self._create_no_op_condensation()

        # Extract Ti, Ai, Oi
        action_event = events[last_action_idx]
        obs_event = events[last_obs_idx]

        thinking = _to_plain_text(getattr(action_event, 'thought', ''))
        observation = _to_plain_text(getattr(obs_event, 'observation', ''))

        # Summarize Ti -> T'i
        summarized_thinking = self._summarize_thinking(thinking)

        # Summarize Oi -> O'i
        summarized_observation = self._summarize_observation(observation)

        # Create condensation that replaces only the current step
        summary = f"[FCM] Condensed step {last_action_idx}: T={len(thinking)} -> {len(summarized_thinking)}, O={len(observation)} -> {len(summarized_observation)}"

        return Condensation(summary=summary)

    def _summarize_thinking(self, text: str) -> str:
        """Summarize thinking if it exceeds threshold"""
        word_count = len(text.split())
        if word_count <= self.thinking_threshold:
            return text

        try:
            prompt = f"Summarize this thinking in 2-3 sentences:\n{text[:2000]}"
            response = litellm.completion(
                model=self.context_llm.model,
                api_key=_unwrap_secret(self.context_llm.api_key),
                base_url=_unwrap_secret(self.context_llm.base_url),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Thinking summarization failed: {e}")
            return text[:200]

    def _summarize_observation(self, text: str) -> str:
        """Summarize observation if it exceeds threshold"""
        word_count = len(text.split())
        if word_count <= self.observation_threshold:
            return text

        try:
            prompt = f"Summarize this observation in 3-4 sentences:\n{text[:2000]}"
            response = litellm.completion(
                model=self.context_llm.model,
                api_key=_unwrap_secret(self.context_llm.api_key),
                base_url=_unwrap_secret(self.context_llm.base_url),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Observation summarization failed: {e}")
            return text[:300]

    def _create_no_op_condensation(self) -> Condensation:
        """Create a no-op condensation when nothing needs to be done"""
        return Condensation(summary="[FCM] No condensation needed")
