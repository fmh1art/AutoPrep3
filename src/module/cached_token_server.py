import json
import itertools
import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from src.tools.funcs import cal_token

logger = logging.getLogger(__name__)


class CachedTokenServer:
    """本地 cached token 计算服务（LRU 淘汰策略）。

    当 LLM 厂商不返回正确的 cached_tokens 时，使用本地前缀匹配计算替代。
    维护最近 max_records 条 LLM 调用记录，以消息粒度进行前缀匹配。

    返回字段：
    - input_tokens: 输入消息的总 token 数
    - output_tokens: 响应消息的 token 数（仅在提供 response_msg 时计算）
    - cached_input_tokens: 通过前缀匹配命中的缓存 token 数
    - uncached_input_tokens: 未命中缓存的输入 token 数（input_tokens - cached_input_tokens）

    淘汰策略：LRU（Least Recently Used），当记录数超过 max_records 时，
    淘汰最近一次被 hit（匹配）时间最早（即最少被使用）的记录。

    性能优化策略：
    - 二级哈希索引（首条消息 / 前两条消息）快速缩小候选范围
    - 消息哈希比较替代完整内容比较
    - 前缀和预计算，匹配后 O(1) 获取 token 数
    - 内容级 token 缓存，避免重复计算
    - 候选按最近优先遍历，尽早找到最长匹配
    """

    def __init__(self, max_records: int = 10000, token_model: str = "cl100k_base"):
        self.max_records = max_records
        self.token_model = token_model

        self._id_counter = itertools.count()

        self._record_map: dict[int, tuple] = {}

        self._first_msg_index: dict[int, set] = defaultdict(set)
        self._prefix2_index: dict[int, set] = defaultdict(set)

        self._token_cache: dict[str, int] = {}

        self._lock = threading.Lock()

    @staticmethod
    def _hash_message(msg: dict) -> int:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content is None:
            content = ""
        tool_calls = msg.get("tool_calls", None)
        tool_call_id = msg.get("tool_call_id", None)
        tc_str = json.dumps(tool_calls, sort_keys=True) if tool_calls else ""
        tc_id_str = tool_call_id or ""
        return hash((role, content, tc_str, tc_id_str))

    def _hash_messages(self, messages: list[dict]) -> list[int]:
        return [self._hash_message(m) for m in messages]

    def _get_token_count(self, content: str) -> int:
        if content in self._token_cache:
            return self._token_cache[content]
        count = cal_token(content, self.token_model)
        self._token_cache[content] = count
        return count

    def _compute_prefix_sums(self, messages: list[dict]) -> list[int]:
        sums = []
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += self._get_token_count(content)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_str = json.dumps(tool_calls, sort_keys=True)
                total += self._get_token_count(tc_str)
            sums.append(total)
        return sums

    def _compute_input_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += self._get_token_count(content)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_str = json.dumps(tool_calls, sort_keys=True)
                total += self._get_token_count(tc_str)
        return total

    def _compute_output_tokens(self, response_msg: dict | str | None) -> int:
        if response_msg is None:
            return 0
        if isinstance(response_msg, str):
            return self._get_token_count(response_msg)
        if isinstance(response_msg, dict):
            content = response_msg.get("content", "") or ""
            tokens = self._get_token_count(content)
            tool_calls = response_msg.get("tool_calls")
            if tool_calls:
                tc_str = json.dumps(tool_calls, sort_keys=True)
                tokens += self._get_token_count(tc_str)
            return tokens
        return 0

    @staticmethod
    def _prefix_match_len(hashes_a: list[int], hashes_b: list[int]) -> int:
        min_len = min(len(hashes_a), len(hashes_b))
        for i in range(min_len):
            if hashes_a[i] != hashes_b[i]:
                return i
        return min_len

    def _touch_record(self, rec_id: int) -> None:
        if rec_id in self._record_map:
            old_entry = self._record_map[rec_id]
            self._record_map[rec_id] = (old_entry[0], old_entry[1], time.time())

    def compute_cached_tokens(self, input_messages: list[dict]) -> dict:
        input_tokens = self._compute_input_tokens(input_messages)

        if not input_messages or not self._record_map:
            return {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "uncached_input_tokens": input_tokens,
            }

        input_hashes = self._hash_messages(input_messages)
        best_match_len = 0
        best_record_id = None

        with self._lock:
            if len(input_hashes) >= 2:
                p2_key = hash((input_hashes[0], input_hashes[1]))
                candidates = sorted(
                    self._prefix2_index.get(p2_key, set()), reverse=True
                )
                for rec_id in candidates:
                    if rec_id not in self._record_map:
                        continue
                    rec_entry = self._record_map[rec_id]
                    rec_hashes = rec_entry[0]
                    if len(rec_hashes) <= best_match_len:
                        continue
                    match_len = self._prefix_match_len(input_hashes, rec_hashes)
                    if match_len > best_match_len:
                        best_match_len = match_len
                        best_record_id = rec_id

            if best_match_len < 2:
                candidates = sorted(
                    self._first_msg_index.get(input_hashes[0], set()), reverse=True
                )
                for rec_id in candidates:
                    if rec_id not in self._record_map:
                        continue
                    rec_entry = self._record_map[rec_id]
                    rec_hashes = rec_entry[0]
                    if len(rec_hashes) <= best_match_len:
                        continue
                    match_len = self._prefix_match_len(input_hashes, rec_hashes)
                    if match_len > best_match_len:
                        best_match_len = match_len
                        best_record_id = rec_id

            if best_match_len == 0:
                recent_candidates = sorted(
                    self._record_map.keys(), reverse=True
                )[:200]
                for rec_id in recent_candidates:
                    rec_entry = self._record_map[rec_id]
                    rec_hashes = rec_entry[0]
                    if len(rec_hashes) <= best_match_len:
                        continue
                    match_len = self._prefix_match_len(input_hashes, rec_hashes)
                    if match_len > best_match_len:
                        best_match_len = match_len
                        best_record_id = rec_id

            cached_input_tokens = 0
            if best_match_len > 0 and best_record_id is not None:
                self._touch_record(best_record_id)
                cached_input_tokens = self._record_map[best_record_id][1][best_match_len - 1]

        uncached_input_tokens = input_tokens - cached_input_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": uncached_input_tokens,
        }

    def cache_conversation(
        self,
        input_messages: list[dict],
        response_msg: dict | str | None = None,
    ) -> dict:
        output_tokens = self._compute_output_tokens(response_msg)

        full_messages = list(input_messages)
        if response_msg is not None:
            if isinstance(response_msg, str):
                full_messages.append({"role": "assistant", "content": response_msg})
            elif isinstance(response_msg, dict):
                full_messages.append(response_msg)

        if not full_messages:
            return {
                "input_tokens": 0,
                "output_tokens": output_tokens,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 0,
            }

        msg_hashes = self._hash_messages(full_messages)
        prefix_sums = self._compute_prefix_sums(full_messages)
        rec_id = next(self._id_counter)

        with self._lock:
            self._record_map[rec_id] = (msg_hashes, prefix_sums, time.time())

            self._first_msg_index[msg_hashes[0]].add(rec_id)

            if len(msg_hashes) >= 2:
                p2_key = hash((msg_hashes[0], msg_hashes[1]))
                self._prefix2_index[p2_key].add(rec_id)

            while len(self._record_map) > self.max_records:
                self._evict_lru()

        return {
            "input_tokens": 0,
            "output_tokens": output_tokens,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
        }

    def query_and_cache(
        self,
        input_messages: list[dict],
        response_msg: dict | str | None = None,
    ) -> dict:
        result = self.compute_cached_tokens(input_messages)
        cache_result = self.cache_conversation(input_messages, response_msg)
        result["output_tokens"] = cache_result["output_tokens"]
        return result

    def _evict_lru(self) -> None:
        if not self._record_map:
            return

        lru_id = None
        lru_time = float("inf")
        for rec_id, entry in self._record_map.items():
            last_hit = entry[2]
            if last_hit < lru_time:
                lru_time = last_hit
                lru_id = rec_id

        if lru_id is None:
            return

        self._remove_record(lru_id)

    def _remove_record(self, rec_id: int) -> None:
        if rec_id not in self._record_map:
            return
        entry = self._record_map.pop(rec_id)
        msg_hashes = entry[0]

        self._first_msg_index[msg_hashes[0]].discard(rec_id)
        if not self._first_msg_index[msg_hashes[0]]:
            del self._first_msg_index[msg_hashes[0]]

        if len(msg_hashes) >= 2:
            p2_key = hash((msg_hashes[0], msg_hashes[1]))
            self._prefix2_index[p2_key].discard(rec_id)
            if not self._prefix2_index[p2_key]:
                del self._prefix2_index[p2_key]

    @property
    def size(self) -> int:
        return len(self._record_map)
