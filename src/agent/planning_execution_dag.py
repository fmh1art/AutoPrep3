"""
PlanningExecutionDAG — 基于依赖图的 Planning-Execution 框架。

与线性 PlanningExecution 的区别：
  1. Planning Agent 在输出每个 operator 时同时给出与之前所有 operator 的依赖度（0~1）
  2. 根据 dependency score 构建有向依赖图（DAG）
  3. 执行时按拓扑排序执行，只有当 op_i 有一条边到 op_j 时，op_i 的信息才会传给 op_j
  4. 信息传输方式为 selective：只传递上游 operator 标记为 useful 的 trajectory steps

本版本为无 replan 版本。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent import CodeAgent
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DAGOperator:
    index: int
    subtask: str
    dependencies: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        deps = {str(k): v for k, v in self.dependencies.items()}
        return {
            "index": self.index,
            "subtask": self.subtask,
            "dependencies": deps,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DAGOperator:
        raw_deps = d.get("dependencies", {})
        deps = {}
        if isinstance(raw_deps, dict):
            for k, v in raw_deps.items():
                try:
                    deps[int(k)] = float(v)
                except (ValueError, TypeError):
                    pass
        return DAGOperator(
            index=d["index"],
            subtask=d["subtask"],
            dependencies=deps,
        )

    def serialize_for_prompt(self) -> str:
        dep_str = ""
        if self.dependencies:
            dep_parts = [f"Op {k} ({v:.1f})" for k, v in sorted(self.dependencies.items())]
            dep_str = f" [depends on: {', '.join(dep_parts)}]"
        return f"[Op {self.index}] {self.subtask}{dep_str}"


@dataclass
class DAGOperatorPlan:
    operators: list[DAGOperator] = field(default_factory=list)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [op.to_dict() for op in self.operators]

    @staticmethod
    def from_dict_list(dl: list[dict[str, Any]]) -> DAGOperatorPlan:
        return DAGOperatorPlan(operators=[DAGOperator.from_dict(d) for d in dl])

    def serialize_for_prompt(self) -> str:
        if not self.operators:
            return "(empty plan)"
        lines = [f"### Operator Plan ({len(self.operators)} operators)", ""]
        for op in self.operators:
            lines.append(op.serialize_for_prompt())
        return "\n".join(lines)

    def reindex(self) -> DAGOperatorPlan:
        for i, op in enumerate(self.operators):
            op.index = i + 1
        return self


@dataclass
class DependencyEdge:
    source: int
    target: int
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "weight": self.weight}


@dataclass
class DependencyGraph:
    operators: list[DAGOperator] = field(default_factory=list)
    edges: list[DependencyEdge] = field(default_factory=list)
    _incoming: dict[int, list[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))
    _outgoing: dict[int, list[DependencyEdge]] = field(default_factory=lambda: defaultdict(list))
    _op_map: dict[int, DAGOperator] = field(default_factory=dict)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self._incoming.clear()
        self._outgoing.clear()
        self._op_map.clear()
        for op in self.operators:
            self._op_map[op.index] = op
        for edge in self.edges:
            self._incoming[edge.target].append(edge)
            self._outgoing[edge.source].append(edge)

    def get_incoming_edges(self, op_index: int) -> list[DependencyEdge]:
        return self._incoming.get(op_index, [])

    def get_outgoing_edges(self, op_index: int) -> list[DependencyEdge]:
        return self._outgoing.get(op_index, [])

    def get_operator(self, op_index: int) -> DAGOperator | None:
        return self._op_map.get(op_index)

    def topological_sort(self) -> list[int]:
        in_degree: dict[int, int] = {op.index: 0 for op in self.operators}
        for edge in self.edges:
            in_degree[edge.target] += 1

        queue = deque()
        for op in self.operators:
            if in_degree[op.index] == 0:
                queue.append(op.index)

        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self._outgoing.get(node, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self.operators):
            logger.warning("Cycle detected in dependency graph, falling back to index order")
            return sorted(self._op_map.keys())

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "operators": [op.to_dict() for op in self.operators],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class RollbackRequest:
    target_operator_indices: list[int] = field(default_factory=list)
    debug_info: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_operator_indices": self.target_operator_indices,
            "debug_info": self.debug_info,
        }


@dataclass
class DAGOperatorExecResult:
    operator_index: int
    operator_subtask: str
    finish_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    hit_max_steps: bool = False
    useful_trajectory_indexes: list[int] = field(default_factory=list)
    trajectory_records: list[dict] = field(default_factory=list)
    rollback_request: RollbackRequest | None = None
    execution_instruction: str = ""
    own_message_start_index: int = 0


ROLLBACK_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "rollback",
        "description": (
            "Call this tool when you discover that a previous operator's work was "
            "incorrect or insufficient, and needs to be re-done with corrective guidance. "
            "This will trigger re-execution of the specified operators with your debug "
            "information, then resume execution from the current operator with updated context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_operators": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "List of operator indices to roll back to. "
                        "These operators will be re-executed in topological order "
                        "with the debug_info provided. If empty, all direct dependency "
                        "operators will be rolled back."
                    ),
                },
                "debug_info": {
                    "type": "string",
                    "description": (
                        "Detailed description of what went wrong and what needs to "
                        "be fixed. This information will be passed to the re-executed "
                        "operators to guide their correction."
                    ),
                },
            },
            "required": ["debug_info"],
        },
    },
}


@dataclass
class DAGPipelineResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: DAGOperatorPlan | None = None
    graph: DependencyGraph | None = None
    exec_results: list[DAGOperatorExecResult] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_dependency_graph(
    plan: DAGOperatorPlan,
    dependency_threshold: float = 0.0,
) -> DependencyGraph:
    edges = []
    for op in plan.operators:
        for dep_idx, score in op.dependencies.items():
            if score >= dependency_threshold:
                edges.append(DependencyEdge(
                    source=dep_idx,
                    target=op.index,
                    weight=score,
                ))

    graph = DependencyGraph(
        operators=list(plan.operators),
        edges=edges,
    )
    logger.info(
        f"Built dependency graph: {len(graph.operators)} nodes, "
        f"{len(graph.edges)} edges (threshold={dependency_threshold})"
    )
    for edge in edges:
        logger.debug(
            f"[DAGGraph] Edge: Op {edge.source} -> Op {edge.target} "
            f"(weight={edge.weight:.2f})"
        )
    dep_matrix = {}
    for op in plan.operators:
        dep_matrix[f"Op_{op.index}"] = {
            f"Op_{k}": round(v, 2) for k, v in sorted(op.dependencies.items())
        }
    logger.debug(f"[DAGGraph] Dependency matrix: {json.dumps(dep_matrix, ensure_ascii=False)}")
    return graph


# ---------------------------------------------------------------------------
# DAG Planning Agent
# ---------------------------------------------------------------------------

class DAGPlanningAgent:
    def __init__(self, llm_cfg: dict):
        self.llm_cfg = llm_cfg
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    @property
    def caller(self):
        return self._caller

    def generate_plan(
        self,
        instruction: str,
        workspace_overview: str = "",
        max_attempts: int = 3,
        output_dir: str | None = None,
    ) -> tuple[DAGOperatorPlan, dict[str, Any]]:
        usage_before = self._caller.get_total_usage()

        user_message = render_j2(
            "planning_agent_dag.j2",
            context={
                "instruction": instruction,
                "workspace_overview": workspace_overview,
            },
        )
        messages = [
            {"role": "user", "content": user_message},
        ]

        planning_output_dir = None
        if output_dir:
            planning_output_dir = os.path.join(output_dir, "planning_agent")
            os.makedirs(planning_output_dir, exist_ok=True)

        for attempt in range(1, max_attempts + 1):
            try:
                attempt_dir = planning_output_dir
                if planning_output_dir and max_attempts > 1:
                    attempt_dir = os.path.join(planning_output_dir, f"attempt_{attempt}")
                    os.makedirs(attempt_dir, exist_ok=True)

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "input.json"), "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=2)

                raw_response = self._caller.chat(messages)
                logger.debug(
                    f"[DAGPlanning] Attempt {attempt} raw response (first 2000 chars): "
                    f"{raw_response[:2000] if raw_response else '(empty)'}"
                )

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "output.txt"), "w", encoding="utf-8") as f:
                        f.write(raw_response if raw_response else "")

                plan = self._parse_plan(raw_response)
                if plan is not None and len(plan.operators) > 0:
                    plan.reindex()
                    self._validate_and_fix_dependencies(plan)
                    logger.info(
                        f"[DAGPlanning] Attempt {attempt} succeeded: "
                        f"{len(plan.operators)} operators parsed"
                    )
                    for op in plan.operators:
                        deps_str = ", ".join(f"Op {k}({v:.2f})" for k, v in sorted(op.dependencies.items()))
                        logger.info(
                            f"[DAGPlanning]   Op {op.index}: {op.subtask[:120]} [deps: {deps_str}]"
                        )
                    if attempt_dir:
                        with open(os.path.join(attempt_dir, "plan.json"), "w", encoding="utf-8") as f:
                            json.dump(plan.to_dict_list(), f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"DAG operator plan generated: {len(plan.operators)} operators"
                    )
                    metrics = self._compute_metrics(usage_before)
                    return plan, metrics
            except Exception as e:
                if attempt_dir:
                    with open(os.path.join(attempt_dir, "error.txt"), "w", encoding="utf-8") as f:
                        f.write(str(e))
                logger.error(f"DAG planning attempt {attempt} failed: {e}")

        logger.warning("All DAG planning attempts failed, using fallback plan")
        metrics = self._compute_metrics(usage_before)
        return self._fallback_plan(), metrics

    def _compute_metrics(self, usage_before: dict) -> dict[str, Any]:
        usage_after = self.caller.get_total_usage()
        return _compute_usage_delta(usage_before, usage_after)

    @staticmethod
    def _validate_and_fix_dependencies(plan: DAGOperatorPlan) -> None:
        valid_indices = {op.index for op in plan.operators}
        for op in plan.operators:
            invalid_deps = [k for k in op.dependencies if k not in valid_indices]
            if invalid_deps:
                logger.warning(
                    f"Op {op.index} has invalid dependency indices: {invalid_deps}. Removing them."
                )
                for k in invalid_deps:
                    del op.dependencies[k]

            forward_deps = [k for k in op.dependencies if k >= op.index]
            if forward_deps:
                logger.warning(
                    f"Op {op.index} has forward dependency indices: {forward_deps}. Removing them."
                )
                for k in forward_deps:
                    del op.dependencies[k]

            for k, v in list(op.dependencies.items()):
                op.dependencies[k] = max(0.0, min(1.0, v))

    def _parse_plan(self, text: str) -> DAGOperatorPlan | None:
        if not text:
            return None

        pattern = r"```json\s*(\[.*?\])\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid = all(
                        isinstance(item, dict)
                        and "subtask" in item
                        for item in parsed
                    )
                    if valid:
                        return DAGOperatorPlan.from_dict_list(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                return DAGOperatorPlan.from_dict_list(parsed)
        except Exception:
            pass

        return None

    @staticmethod
    def _fallback_plan() -> DAGOperatorPlan:
        return DAGOperatorPlan(operators=[
            DAGOperator(
                index=1,
                subtask="Complete the entire task end-to-end: explore the codebase, understand the issue, implement the fix, and verify it works",
                dependencies={},
            ),
        ])


# ---------------------------------------------------------------------------
# DAG-specific Constants
# ---------------------------------------------------------------------------

DAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer.\n\n"
    "This workflow should be done step-by-step so that you can iterate on your changes "
    "and any possible problems. Be thorough in your exploration, testing, and reasoning. "
    "It's fine if your thinking process is lengthy - quality and completeness are more "
    "important than brevity."
)

DAG_SEPARATOR = (
    "---\n"
    "The above context is from previously executed operators and remains unchanged. "
    "Now proceed to your assigned operator.\n"
    "---"
)


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "prompt_tokens", "completion_tokens", "cache_read_tokens",
    "reasoning_tokens", "total_tokens",
)


def _compute_usage_delta(usage_before: dict, usage_after: dict) -> dict[str, Any]:
    return {
        "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
        "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
        "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
        "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
        "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
        "accumulated_cost": 0.0,
    }


def _compute_usage_delta_from_agent(usage_before: dict, usage_after: dict) -> dict[str, Any]:
    key_map = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "cache_read_tokens": "cached_tokens",
        "reasoning_tokens": "reasoning_tokens",
        "total_tokens": "total_tokens",
    }
    return {
        k: usage_after.get(v, 0) - usage_before.get(v, 0)
        for k, v in key_map.items()
    } | {"accumulated_cost": 0.0}


def _parse_agent_other_content(result) -> tuple[str, list[int], RollbackRequest | None, list[dict]]:
    if not hasattr(result, 'other_content') or not isinstance(result.other_content, dict):
        return "", [], None, []

    oc = result.other_content
    finish_msg = oc.get("finish_message", "")

    raw_indexes = oc.get("useful_trajectory_indexes", [])
    useful_indexes = sorted(i for i in raw_indexes if isinstance(i, int))

    rollback_req = None
    if oc.get("terminated_tool") == "rollback":
        tool_args = oc.get("terminated_tool_args", {})
        rollback_req = RollbackRequest(
            target_operator_indices=tool_args.get("target_operators", []),
            debug_info=tool_args.get("debug_info", ""),
        )

    trajectory_records = oc.get("trajectory_records", [])
    return finish_msg, useful_indexes, rollback_req, trajectory_records


def _build_dependency_info(operator: DAGOperator, plan: DAGOperatorPlan, threshold: float) -> list[dict]:
    op_map = {op.index: op for op in plan.operators}
    dep_info = []
    for dep_idx, score in sorted(operator.dependencies.items()):
        if score >= threshold:
            dep_op = op_map.get(dep_idx)
            if dep_op:
                dep_info.append({"index": dep_idx, "subtask": dep_op.subtask, "score": score})
    return dep_info


def _sum_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    total = {k: 0 for k in _METRIC_KEYS} | {"accumulated_cost": 0.0}
    for m in metrics_list:
        for k in _METRIC_KEYS:
            total[k] += m.get(k, 0)
        total["accumulated_cost"] += m.get("accumulated_cost", 0.0)
    return total


def _merge_metrics_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    result = {k: 0 for k in _METRIC_KEYS} | {"accumulated_cost": 0.0}
    for d in dicts:
        for k in _METRIC_KEYS:
            result[k] += d.get(k, 0)
        result["accumulated_cost"] += d.get("accumulated_cost", 0.0)
    return result


# ---------------------------------------------------------------------------
# DAG Operator Executor
# ---------------------------------------------------------------------------

class DAGOperatorExecutor:

    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_operator: int = 80,
        max_retries_per_call: int = 3,
        selective_fallback_rule: str = "all",
        use_optimized_agent: bool = False,
        max_time_per_operator: float | None = None,
        max_index_validation_retries: int = 2,
        dependency_threshold: float = 0.1,
        enable_rollback: bool = True,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps_per_operator = max_steps_per_operator
        self.max_retries_per_call = max_retries_per_call
        self.selective_fallback_rule = selective_fallback_rule
        self.use_optimized_agent = use_optimized_agent
        self.max_time_per_operator = max_time_per_operator
        self.max_index_validation_retries = max_index_validation_retries
        self.dependency_threshold = dependency_threshold
        self.enable_rollback = enable_rollback

        self._agent = self._create_agent()

    def _create_agent(self):
        extra_tools = [ROLLBACK_TOOL_DEFINITION] if self.enable_rollback else []
        terminating = ["rollback"] if self.enable_rollback else []

        if self.use_optimized_agent:
            from src.agent.code_agent_optimized import CodeAgentOptimized, TOOL_DEFINITIONS as OPT_TOOL_DEFS
            tools = list(OPT_TOOL_DEFS) + extra_tools
            return CodeAgentOptimized(
                llm_cfg=self.llm_cfg,
                max_steps=self.max_steps_per_operator,
                max_retries_per_call=self.max_retries_per_call,
                include_steps=False,
                max_time=self.max_time_per_operator,
                tool_definitions=tools,
                terminating_tools=terminating,
                system_prompt=DAG_SYSTEM_PROMPT,
            )
        from src.agent.code_agent import TOOL_DEFINITIONS as BASE_TOOL_DEFS
        tools = list(BASE_TOOL_DEFS) + extra_tools
        return CodeAgent(
            llm_cfg=self.llm_cfg,
            max_steps=self.max_steps_per_operator,
            max_retries_per_call=self.max_retries_per_call,
            include_steps=False,
            max_time=self.max_time_per_operator,
            tool_definitions=tools,
            terminating_tools=terminating,
            system_prompt=DAG_SYSTEM_PROMPT,
        )

    # ---- Selective mode helpers ----

    @staticmethod
    def _validate_useful_indexes(
        raw_indexes: list[int],
        trajectory_offset: int,
        total_steps: int,
    ) -> list[int]:
        validated = []
        for idx in raw_indexes:
            if idx < trajectory_offset:
                continue
            if idx >= trajectory_offset + total_steps:
                continue
            validated.append(idx)
        return sorted(set(validated))

    @staticmethod
    def _apply_fallback_rule(
        total_steps: int,
        trajectory_offset: int,
        rule: str,
    ) -> list[int]:
        if total_steps == 0:
            return []

        if rule == "all":
            return list(range(trajectory_offset, trajectory_offset + total_steps))
        elif rule == "last_half":
            count = max(1, total_steps // 2)
        elif rule == "last_third":
            count = max(1, total_steps // 3)
        elif rule == "none":
            return []
        else:
            count = max(1, total_steps // 2)

        start = trajectory_offset + total_steps - count
        return list(range(start, trajectory_offset + total_steps))

    @staticmethod
    def _filter_messages_by_indexes(
        messages: list[dict],
        useful_indexes: list[int],
        include_user_messages: bool = False,
    ) -> list[dict]:
        if not useful_indexes:
            return []

        index_set = set(useful_indexes)
        useful_tool_call_ids: set[str] = set()
        filtered = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                continue

            if role == "assistant":
                DAGOperatorExecutor._filter_assistant_msg(
                    msg, index_set, useful_tool_call_ids, filtered
                )
            elif role == "tool":
                DAGOperatorExecutor._filter_tool_msg(
                    msg, useful_tool_call_ids, filtered
                )
            elif role == "user" and include_user_messages:
                content = msg.get("content", "")
                if content and "Please continue working" not in content:
                    filtered.append({"role": "user", "content": content})

        return filtered

    @staticmethod
    def _filter_assistant_msg(
        msg: dict,
        index_set: set[int],
        useful_tool_call_ids: set[str],
        filtered: list[dict],
    ):
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            matching_tcs = []
            for tc in tool_calls:
                tc_fn = tc.get("function", {})
                tc_name = tc_fn.get("name", "")
                if tc_name == "finish":
                    continue

                tc_args_str = tc_fn.get("arguments", "{}")
                try:
                    tc_args = json.loads(tc_args_str)
                except (json.JSONDecodeError, TypeError):
                    tc_args = {}

                step_idx = tc_args.get("_trajectory_step_index")
                if step_idx is not None and step_idx in index_set:
                    clean_tc = {
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                        "function": {
                            "name": tc_name,
                            "arguments": tc_fn.get("arguments", "{}"),
                        },
                    }
                    matching_tcs.append(clean_tc)
                    useful_tool_call_ids.add(tc.get("id"))

            if matching_tcs:
                filtered_msg = {"role": "assistant"}
                content = msg.get("content")
                if content:
                    filtered_msg["content"] = content
                filtered_msg["tool_calls"] = matching_tcs
                filtered.append(filtered_msg)
        else:
            content = msg.get("content") or ""
            if content:
                filtered.append({"role": "assistant", "content": content})

    @staticmethod
    def _filter_tool_msg(
        msg: dict,
        useful_tool_call_ids: set[str],
        filtered: list[dict],
    ):
        tc_id = msg.get("tool_call_id")
        if not tc_id or tc_id not in useful_tool_call_ids:
            return
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        filtered.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": content,
        })

    # ---- File extraction ----

    @staticmethod
    def _parse_tool_args(tool_args: dict | str) -> dict:
        if isinstance(tool_args, dict):
            if "_raw" in tool_args:
                try:
                    return json.loads(tool_args["_raw"])
                except (json.JSONDecodeError, TypeError):
                    return tool_args
            return tool_args
        if isinstance(tool_args, str):
            try:
                return json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @staticmethod
    def _compute_total_steps(messages: list[dict]) -> int:
        max_step = -1
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        idx = args.get("_trajectory_step_index")
                        if idx is not None and isinstance(idx, int):
                            max_step = max(max_step, idx)
                    except (json.JSONDecodeError, TypeError):
                        pass
        return max_step + 1 if max_step >= 0 else 0

    @staticmethod
    def _extract_files_from_trajectory(messages: list[dict]) -> tuple[list[str], list[str]]:
        modified = set()
        read = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("tool_name", "")
            raw_args = msg.get("tool_args", {})
            tool_args = DAGOperatorExecutor._parse_tool_args(raw_args)

            if tool_name == "file_editor":
                path = tool_args.get("path", "")
                command = tool_args.get("command", "")
                if not path:
                    continue
                if command in ("str_replace", "insert", "create"):
                    modified.add(path)
                elif command == "view":
                    read.add(path)
            elif tool_name == "string_replace":
                path = tool_args.get("path", "")
                if path:
                    modified.add(path)
            elif tool_name in ("view_file", "search_by_keyword"):
                path = tool_args.get("path", "")
                if path:
                    read.add(path)
            elif tool_name in ("bash", "terminal"):
                cmd = tool_args.get("command", "")
                for match in re.finditer(r'(?:cat|head|tail|less|more)\s+(\S+)', cmd):
                    read.add(match.group(1))

        return sorted(modified), sorted(read)

    # ---- Collect context from dependencies ----

    def _collect_dependency_context(
        self,
        operator: DAGOperator,
        graph: DependencyGraph,
        exec_results_map: dict[int, DAGOperatorExecResult],
        instruction: str = "",
        repo_path: str = "/workspace",
    ) -> tuple[list[dict] | None, list[dict] | None]:
        incoming_edges = graph.get_incoming_edges(operator.index)
        significant_edges = [
            e for e in incoming_edges if e.weight >= self.dependency_threshold
        ]

        if not significant_edges:
            logger.debug(
                f"[DAGContext] Op {operator.index}: no significant incoming edges "
                f"(incoming={len(incoming_edges)}, significant=0)"
            )
            return None, None

        sorted_edges = sorted(significant_edges, key=lambda e: e.source)
        initial_messages = self._build_initial_messages(
            sorted_edges, exec_results_map, instruction, repo_path,
        )
        prefix_trajectory = self._build_prefix_trajectory(sorted_edges, exec_results_map)

        if initial_messages:
            logger.debug(
                f"[DAGContext] Op {operator.index}: selective mode, "
                f"passing {len(initial_messages)} filtered messages"
            )
        if prefix_trajectory:
            logger.debug(
                f"[DAGContext] Op {operator.index}: selective mode, "
                f"passing {len(prefix_trajectory)} prefix trajectory records"
            )

        return initial_messages, prefix_trajectory

    def _build_initial_messages(
        self,
        sorted_edges: list[DependencyEdge],
        exec_results_map: dict[int, DAGOperatorExecResult],
        instruction: str,
        repo_path: str,
    ) -> list[dict] | None:
        task_query = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        all_filtered = [
            {"role": "system", "content": DAG_SYSTEM_PROMPT},
            {"role": "user", "content": task_query},
        ]

        for edge in sorted_edges:
            dep_result = exec_results_map.get(edge.source)
            if dep_result is None or not dep_result.messages:
                continue

            own_messages = dep_result.messages[dep_result.own_message_start_index:]
            filtered = self._filter_dep_messages(dep_result, own_messages)

            if not filtered:
                continue

            transition_content = (
                dep_result.execution_instruction
                if dep_result.execution_instruction
                else (
                    f"--- Continuing from Operator {edge.source} "
                    f"(dependency: {edge.weight:.1f}): "
                    f"{dep_result.operator_subtask} ---\n\n"
                    f"Previous operator summary: "
                    f"{dep_result.finish_message[:500] if dep_result.finish_message else 'N/A'}"
                )
            )
            all_filtered.append({"role": "user", "content": transition_content})
            all_filtered.extend(filtered)

        return all_filtered if len(all_filtered) > 2 else None

    def _filter_dep_messages(
        self,
        dep_result: DAGOperatorExecResult,
        own_messages: list[dict],
    ) -> list[dict]:
        if dep_result.useful_trajectory_indexes:
            indexes = dep_result.useful_trajectory_indexes
        else:
            total_steps = self._compute_total_steps(own_messages)
            indexes = self._apply_fallback_rule(total_steps, 0, self.selective_fallback_rule)
        return self._filter_messages_by_indexes(own_messages, indexes, include_user_messages=False)

    def _build_prefix_trajectory(
        self,
        significant_edges: list[DependencyEdge],
        exec_results_map: dict[int, DAGOperatorExecResult],
    ) -> list[dict] | None:
        prefix_records = []
        for edge in significant_edges:
            dep_result = exec_results_map.get(edge.source)
            if dep_result is None or not dep_result.trajectory_records:
                continue

            useful_idx = set(dep_result.useful_trajectory_indexes)
            useful_records = [
                rec for rec in dep_result.trajectory_records
                if rec.get("role") in ("tool", "assistant")
                and rec.get("index", -999) in useful_idx
            ]
            if useful_records:
                prefix_records.append({
                    "index": edge.source,
                    "role": "transition",
                    "content": (
                        f"Operator {edge.source} completed subtask: "
                        f"{dep_result.operator_subtask[:200]}\n\n"
                        f"Summary: {dep_result.finish_message[:500] if dep_result.finish_message else 'N/A'}"
                    ),
                })
                prefix_records.extend(useful_records)

        return prefix_records if prefix_records else None

    # ---- Validate & fallback useful indexes ----

    def _resolve_useful_indexes(
        self,
        useful_indexes: list[int],
        total_steps: int,
        validation_attempt: int,
        max_validation_retries: int,
        operator_index: int,
        label: str = "",
    ) -> tuple[list[int], bool]:
        invalid_indexes = [idx for idx in useful_indexes if idx < 0 or idx >= total_steps]
        if invalid_indexes:
            if validation_attempt < max_validation_retries:
                logger.warning(
                    f"  [Op {operator_index}]{label} Invalid useful_trajectory_indexes: "
                    f"{invalid_indexes}. Retrying..."
                )
                return useful_indexes, True

            useful_indexes = [idx for idx in useful_indexes if 0 <= idx < total_steps]
            if not useful_indexes:
                useful_indexes = self._apply_fallback_rule(
                    total_steps, 0, self.selective_fallback_rule,
                )

        if not useful_indexes:
            logger.warning(
                f"  [selective] Operator {operator_index}{label} did not provide "
                f"useful_trajectory_indexes, applying fallback rule: "
                f"{self.selective_fallback_rule}"
            )
            useful_indexes = self._apply_fallback_rule(
                total_steps, 0, self.selective_fallback_rule,
            )

        return useful_indexes, False

    # ---- Execute a single operator (unified for normal & rollback) ----

    def execute_operator(
        self,
        operator: DAGOperator,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        initial_messages: list[dict] | None = None,
        dependency_info: list[dict] | None = None,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        prefix_trajectory: list[dict] | None = None,
        rollback_context: dict | None = None,
    ) -> DAGOperatorExecResult:
        dep_info = dependency_info or []
        is_rollback = rollback_context is not None
        label = " rollback" if is_rollback else ""

        exec_prompt = self._render_exec_prompt(
            operator, plan, dep_info, rollback_context,
        )

        logger.info(
            f"{'[DAGRollback] Re-executing' if is_rollback else 'Executing'} operator "
            f"{operator.index}/{len(plan.operators)} "
            f"(deps={list(operator.dependencies.keys())}): "
            f"{operator.subtask[:100]}"
        )
        logger.debug(
            f"[DAGExec] Op {operator.index}{label} exec_prompt (first 1500 chars): "
            f"{exec_prompt[:1500]}"
        )
        if dep_info:
            logger.debug(
                f"[DAGExec] Op {operator.index}{label} dependency_info: "
                f"{json.dumps(dep_info, ensure_ascii=False)}"
            )

        op_output_dir = self._make_op_output_dir(output_dir, operator.index, is_rollback)

        try:
            if initial_messages:
                initial_messages = list(initial_messages) + [
                    {"role": "user", "content": DAG_SEPARATOR},
                    {"role": "user", "content": exec_prompt},
                ]
                logger.info(
                    f"  [Op {operator.index}{label}] Passing {len(initial_messages)} messages "
                    f"(system + task_query + dependencies + separator + current prompt)"
                )
            else:
                task_query = render_j2("query.j2", context={
                    "problem_statement": instruction,
                    "repo_path": repo_path,
                })
                initial_messages = [
                    {"role": "system", "content": DAG_SYSTEM_PROMPT},
                    {"role": "user", "content": task_query},
                    {"role": "user", "content": exec_prompt},
                ]
                logger.info(
                    f"  [Op {operator.index}{label}] No dependency context (fresh start)"
                )

            own_message_start = len(initial_messages) - 1
            result, op_metrics, finish_msg, useful_indexes, rollback_req, trajectory_records = \
                self._run_agent_with_validation(
                    exec_prompt, workspace, callbacks, op_output_dir or output_dir,
                    initial_messages, prefix_trajectory, own_message_start,
                    operator.index, label,
                )

            result_messages = result.messages if hasattr(result, 'messages') else []
            hit_max_steps = len(result_messages) >= self.max_steps_per_operator * 2
            if not finish_msg and result_messages:
                finish_msg = "(operator finished without explicit message)"

            files_modified, files_read = self._extract_files_from_trajectory(result_messages)

            logger.info(
                f"  Operator {operator.index}{label} done: "
                f"tokens={op_metrics['total_tokens']}, "
                f"messages={len(result_messages)}, "
                f"finish_msg_len={len(finish_msg)}, "
                f"files_modified={len(files_modified)}, "
                f"hit_max_steps={hit_max_steps}"
            )
            logger.debug(
                f"[DAGExec] Op {operator.index}{label} finish_message: "
                f"{finish_msg[:500] if finish_msg else '(empty)'}"
            )
            logger.debug(
                f"[DAGExec] Op {operator.index}{label} useful_trajectory_indexes: "
                f"{useful_indexes}"
            )
            if files_modified:
                logger.debug(f"[DAGExec] Op {operator.index}{label} files_modified: {files_modified}")
            if files_read:
                logger.debug(f"[DAGExec] Op {operator.index}{label} files_read: {files_read}")

            return DAGOperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                finish_message=finish_msg,
                metrics=op_metrics,
                messages=result_messages,
                files_modified=files_modified,
                files_read=files_read,
                hit_max_steps=hit_max_steps,
                useful_trajectory_indexes=useful_indexes,
                trajectory_records=trajectory_records,
                rollback_request=rollback_req,
                execution_instruction=exec_prompt,
                own_message_start_index=own_message_start,
            )

        except Exception as e:
            logger.error(f"Operator {operator.index}{label} execution failed: {e}")
            return DAGOperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                error=str(e),
            )

    def _render_exec_prompt(
        self,
        operator: DAGOperator,
        plan: DAGOperatorPlan,
        dep_info: list[dict],
        rollback_context: dict | None,
    ) -> str:
        base_context = {
            "operator_index": operator.index,
            "total_operators": len(plan.operators),
            "operator_subtask": operator.subtask,
            "dependencies": dep_info,
            "is_selective": True,
        }

        if rollback_context is None:
            return render_j2("operator_execution_dag.j2", context=base_context)

        return render_j2(
            "operator_execution_dag_rollback.j2",
            context=base_context | rollback_context,
        )

    def _make_op_output_dir(
        self, output_dir: str | None, op_index: int, is_rollback: bool,
    ) -> str | None:
        if not output_dir:
            return None
        suffix = f"operator_{op_index}_rollback" if is_rollback else f"operator_{op_index}"
        op_dir = os.path.join(output_dir, suffix)
        os.makedirs(op_dir, exist_ok=True)
        return op_dir

    def _run_agent_with_validation(
        self,
        exec_prompt: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None,
        output_dir: str | None,
        initial_messages: list[dict] | None,
        prefix_trajectory: list[dict] | None,
        own_message_start: int,
        operator_index: int,
        label: str,
    ) -> tuple:
        max_validation_retries = self.max_index_validation_retries

        for validation_attempt in range(max_validation_retries + 1):
            if validation_attempt > 0:
                logger.info(
                    f"  [Op {operator_index}{label}] Retrying "
                    f"(attempt {validation_attempt + 1}/{max_validation_retries + 1})..."
                )

            usage_before = self._agent.caller.get_total_usage()
            result = self._agent.run(
                instruction=exec_prompt,
                workspace=workspace,
                callbacks=callbacks,
                output_dir=output_dir,
                initial_messages=initial_messages,
                prefix_trajectory=prefix_trajectory,
            )
            usage_after = self._agent.caller.get_total_usage()

            op_metrics = _compute_usage_delta_from_agent(usage_before, usage_after)
            finish_msg, useful_indexes, rollback_req, trajectory_records = \
                _parse_agent_other_content(result)

            result_messages = result.messages if hasattr(result, 'messages') else []
            own_messages = result_messages[own_message_start:]
            total_steps = self._compute_total_steps(own_messages)

            useful_indexes, should_retry = self._resolve_useful_indexes(
                useful_indexes, total_steps, validation_attempt,
                max_validation_retries, operator_index, label,
            )
            if should_retry:
                continue

            return result, op_metrics, finish_msg, useful_indexes, rollback_req, trajectory_records

    # ---- Rollback helpers ----

    def _compute_rollback_targets(
        self,
        rollback_request: RollbackRequest,
        operator: DAGOperator,
        graph: DependencyGraph,
    ) -> list[int]:
        if rollback_request.target_operator_indices:
            specified = set(rollback_request.target_operator_indices)
            valid = {op.index for op in graph.operators}
            targets = sorted(specified & valid)
        else:
            incoming_edges = graph.get_incoming_edges(operator.index)
            significant_edges = [
                e for e in incoming_edges if e.weight >= self.dependency_threshold
            ]
            targets = sorted({e.source for e in significant_edges})

        topo_order = graph.topological_sort()
        topo_rank = {idx: rank for rank, idx in enumerate(topo_order)}
        targets.sort(key=lambda idx: topo_rank.get(idx, 0))

        logger.info(
            f"[DAGRollback] Computed rollback targets for Op {operator.index}: "
            f"{targets} (topo order)"
        )
        return targets

    def _re_execute_operator(
        self,
        operator: DAGOperator,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str,
        previous_result: DAGOperatorExecResult,
        rollback_request: RollbackRequest,
        exec_results_map: dict[int, DAGOperatorExecResult],
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        all_rollback_targets: list[int] | None = None,
    ) -> DAGOperatorExecResult:
        initial_messages, prefix_trajectory = \
            self._collect_dependency_context(
                operator=operator,
                graph=graph,
                exec_results_map=exec_results_map,
                instruction=instruction,
                repo_path=repo_path,
            )

        dep_info = _build_dependency_info(operator, plan, self.dependency_threshold)

        previous_execution_summary = (
            f"Finish message: {previous_result.finish_message}\n"
            f"Files modified: {', '.join(previous_result.files_modified) or 'None'}\n"
            f"Files read: {', '.join(previous_result.files_read) or 'None'}"
        )

        rollback_source = (
            rollback_request.target_operator_indices[0]
            if rollback_request.target_operator_indices
            else operator.index
        )
        rollback_targets = all_rollback_targets or rollback_request.target_operator_indices or [operator.index]

        rollback_context = {
            "debug_info": rollback_request.debug_info,
            "previous_execution_summary": previous_execution_summary,
            "rollback_source_operator": rollback_source,
            "all_rollback_targets": rollback_targets,
            "is_direct_target": operator.index in rollback_targets,
        }

        return self.execute_operator(
            operator=operator,
            plan=plan,
            graph=graph,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            initial_messages=initial_messages,
            dependency_info=dep_info,
            callbacks=callbacks,
            output_dir=output_dir,
            prefix_trajectory=prefix_trajectory,
            rollback_context=rollback_context,
        )

    # ---- Execute the full DAG plan ----

    def execute_plan(
        self,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        max_rollback_attempts: int = 3,
    ) -> tuple[list[DAGOperatorExecResult], dict[str, Any]]:
        results: list[DAGOperatorExecResult] = []
        exec_results_map: dict[int, DAGOperatorExecResult] = {}

        op_map = {op.index: op for op in plan.operators}

        execution_order = graph.topological_sort()
        logger.info(f"DAG execution order: {execution_order}")
        logger.debug(
            f"[DAGExec] Full execution order with subtask names: "
            + json.dumps([
                {"op": idx, "subtask": op_map[idx].subtask[:80]}
                for idx in execution_order if idx in op_map
            ], ensure_ascii=False)
        )

        rollback_attempts_map: dict[int, int] = defaultdict(int)
        rollback_history: list[dict[str, Any]] = []

        exec_idx = 0
        while exec_idx < len(execution_order):
            op_index = execution_order[exec_idx]
            operator = op_map.get(op_index)
            if operator is None:
                logger.warning(f"Operator {op_index} not found in plan, skipping")
                exec_idx += 1
                continue

            result = self._execute_single_operator(
                operator, plan, graph, instruction, workspace,
                repo_path, callbacks, output_dir, exec_results_map,
            )
            results.append(result)
            exec_results_map[operator.index] = result

            logger.debug(
                f"[DAGExec] Op {operator.index} result: "
                f"error={result.error}, "
                f"finish_msg_len={len(result.finish_message)}, "
                f"files_modified={result.files_modified}, "
                f"files_read={result.files_read}, "
                f"hit_max_steps={result.hit_max_steps}, "
                f"useful_indexes={result.useful_trajectory_indexes}, "
                f"tokens={result.metrics.get('total_tokens', 0)}"
            )

            if output_dir:
                self._save_operator_result(output_dir, operator, result)

            if result.rollback_request is not None and self.enable_rollback:
                self._handle_rollback(
                    result, operator, plan, graph, instruction, workspace,
                    repo_path, callbacks, output_dir, op_map, execution_order,
                    exec_results_map, results, rollback_attempts_map,
                    rollback_history, max_rollback_attempts,
                )

            exec_idx += 1

        total_metrics = self.compute_total_metrics(results)
        if rollback_history:
            total_metrics["rollback_history"] = rollback_history
        return results, total_metrics

    def _execute_single_operator(
        self,
        operator: DAGOperator,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str,
        callbacks: list[Callable] | None,
        output_dir: str | None,
        exec_results_map: dict[int, DAGOperatorExecResult],
    ) -> DAGOperatorExecResult:
        initial_messages, prefix_trajectory = \
            self._collect_dependency_context(
                operator=operator,
                graph=graph,
                exec_results_map=exec_results_map,
                instruction=instruction,
                repo_path=repo_path,
            )

        dep_info = _build_dependency_info(operator, plan, self.dependency_threshold)

        return self.execute_operator(
            operator=operator,
            plan=plan,
            graph=graph,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            initial_messages=initial_messages,
            dependency_info=dep_info,
            callbacks=callbacks,
            output_dir=output_dir,
            prefix_trajectory=prefix_trajectory,
        )

    def _handle_rollback(
        self,
        result: DAGOperatorExecResult,
        operator: DAGOperator,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str,
        callbacks: list[Callable] | None,
        output_dir: str | None,
        op_map: dict[int, DAGOperator],
        execution_order: list[int],
        exec_results_map: dict[int, DAGOperatorExecResult],
        results: list[DAGOperatorExecResult],
        rollback_attempts_map: dict[int, int],
        rollback_history: list[dict[str, Any]],
        max_rollback_attempts: int,
    ):
        rollback_req = result.rollback_request
        total_rollback_attempts = sum(rollback_attempts_map.values())

        if total_rollback_attempts >= max_rollback_attempts:
            logger.warning(
                f"[DAGRollback] Max total rollback attempts ({max_rollback_attempts}) "
                f"reached. Continuing without rollback."
            )
            return

        targets = self._compute_rollback_targets(
            rollback_request=rollback_req,
            operator=operator,
            graph=graph,
        )

        if not targets:
            logger.warning(
                f"[DAGRollback] No valid rollback targets for Op {operator.index}. "
                f"Continuing without rollback."
            )
            return

        logger.info(
            f"[DAGRollback] Op {operator.index} requests rollback to {targets} "
            f"with debug_info: {rollback_req.debug_info[:200]}"
        )

        rollback_history.append({
            "source_operator": operator.index,
            "target_operators": targets,
            "debug_info": rollback_req.debug_info,
            "rollback_attempt": total_rollback_attempts + 1,
        })

        for target_idx in targets:
            target_op = op_map.get(target_idx)
            if target_op is None:
                logger.warning(f"[DAGRollback] Target Op {target_idx} not found, skipping")
                continue

            previous_target_result = exec_results_map.get(target_idx)
            if previous_target_result is None:
                logger.warning(
                    f"[DAGRollback] No previous result for Op {target_idx}, skipping"
                )
                continue

            if rollback_attempts_map[target_idx] >= max_rollback_attempts:
                logger.warning(
                    f"[DAGRollback] Op {target_idx} has reached max rollback attempts "
                    f"({max_rollback_attempts}). Skipping."
                )
                continue

            rollback_attempts_map[target_idx] += 1
            new_result = self._re_execute_operator(
                operator=target_op,
                plan=plan,
                graph=graph,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                previous_result=previous_target_result,
                rollback_request=rollback_req,
                exec_results_map=exec_results_map,
                callbacks=callbacks,
                output_dir=output_dir,
                all_rollback_targets=targets,
            )
            results.append(new_result)
            exec_results_map[target_idx] = new_result

            logger.info(
                f"[DAGRollback] Op {target_idx} re-executed: "
                f"error={new_result.error}, "
                f"finish_msg_len={len(new_result.finish_message)}, "
                f"files_modified={new_result.files_modified}"
            )

            if output_dir:
                self._save_operator_result(output_dir, target_op, new_result)

        self._re_execute_downstream(
            targets, operator.index, plan, graph, instruction, workspace,
            repo_path, callbacks, output_dir, op_map, execution_order,
            exec_results_map, results,
        )

    def _re_execute_downstream(
        self,
        rollback_targets: list[int],
        source_op_index: int,
        plan: DAGOperatorPlan,
        graph: DependencyGraph,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str,
        callbacks: list[Callable] | None,
        output_dir: str | None,
        op_map: dict[int, DAGOperator],
        execution_order: list[int],
        exec_results_map: dict[int, DAGOperatorExecResult],
        results: list[DAGOperatorExecResult],
    ):
        downstream_ops = self._get_downstream_ops(
            rollback_targets, graph, execution_order, source_op_index,
        )
        for ds_idx in downstream_ops:
            ds_op = op_map.get(ds_idx)
            if ds_op is None:
                continue
            previous_ds_result = exec_results_map.get(ds_idx)
            if previous_ds_result is None:
                continue

            logger.info(
                f"[DAGRollback] Re-executing downstream Op {ds_idx} "
                f"(affected by rollback to {rollback_targets})"
            )

            ds_rollback_req = RollbackRequest(
                target_operator_indices=[],
                debug_info=(
                    f"Upstream operators {rollback_targets} were re-executed due to a rollback. "
                    f"Your context has been updated. Please re-execute your subtask "
                    f"with the updated information from the corrected upstream operators."
                ),
            )

            new_ds_result = self._re_execute_operator(
                operator=ds_op,
                plan=plan,
                graph=graph,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                previous_result=previous_ds_result,
                rollback_request=ds_rollback_req,
                exec_results_map=exec_results_map,
                callbacks=callbacks,
                output_dir=output_dir,
                all_rollback_targets=rollback_targets,
            )
            results.append(new_ds_result)
            exec_results_map[ds_idx] = new_ds_result

            if output_dir:
                self._save_operator_result(output_dir, ds_op, new_ds_result)

    @staticmethod
    def _get_downstream_ops(
        rollback_targets: list[int],
        graph: DependencyGraph,
        execution_order: list[int],
        source_op_index: int,
    ) -> list[int]:
        downstream = set()
        for target_idx in rollback_targets:
            for op_idx in execution_order:
                if op_idx <= target_idx or op_idx == source_op_index:
                    continue
                incoming = graph.get_incoming_edges(op_idx)
                dep_indices = {e.source for e in incoming}
                if dep_indices & (set(rollback_targets) | downstream) or target_idx in dep_indices:
                    downstream.add(op_idx)

        return [idx for idx in execution_order if idx in downstream]

    def _save_operator_result(
        self,
        output_dir: str,
        operator: DAGOperator,
        result: DAGOperatorExecResult,
    ):
        op_dir = os.path.join(output_dir, f"operator_{operator.index}")
        os.makedirs(op_dir, exist_ok=True)

        result_data = {
            "operator_index": result.operator_index,
            "operator_subtask": result.operator_subtask,
            "finish_message": result.finish_message,
            "metrics": result.metrics,
            "error": result.error,
            "files_modified": result.files_modified,
            "files_read": result.files_read,
            "hit_max_steps": result.hit_max_steps,
            "useful_trajectory_indexes": result.useful_trajectory_indexes,
            "own_message_start_index": result.own_message_start_index,
            "dependencies": operator.dependencies,
        }

        with open(os.path.join(op_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        if result.messages:
            try:
                messages_data = {
                    "operator_index": result.operator_index,
                    "operator_subtask": result.operator_subtask,
                    "messages_count": len(result.messages),
                    "messages": result.messages,
                }
                with open(os.path.join(op_dir, "messages.json"), "w", encoding="utf-8") as f:
                    json.dump(messages_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save messages for operator {operator.index}: {e}")

        if result.trajectory_records:
            try:
                trajectory_data = {
                    "operator_index": result.operator_index,
                    "operator_subtask": result.operator_subtask,
                    "total_steps": len(result.trajectory_records),
                    "useful_trajectory_indexes": result.useful_trajectory_indexes,
                    "trajectory": result.trajectory_records,
                }
                with open(os.path.join(op_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                    json.dump(trajectory_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save trajectory.json for operator {operator.index}: {e}")

    @staticmethod
    def compute_total_metrics(results: list[DAGOperatorExecResult]) -> dict[str, Any]:
        return {"total": _sum_metrics([r.metrics for r in results])}


# ---------------------------------------------------------------------------
# DAG PlanningExecution Pipeline
# ---------------------------------------------------------------------------

class DAGPlanningExecutionPipeline:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_operator: int = 80,
        selective_fallback_rule: str = "all",
        use_optimized_agent: bool = True,
        max_time_per_operator: float | None = None,
        max_index_validation_retries: int = 2,
        dependency_threshold: float = 0.1,
        max_rollback_attempts: int = 3,
        enable_rollback: bool = True,
    ):
        self.dependency_threshold = dependency_threshold
        self.max_rollback_attempts = max_rollback_attempts
        self.enable_rollback = enable_rollback

        self.planning_agent = DAGPlanningAgent(llm_cfg=llm_cfg)

        self.executor = DAGOperatorExecutor(
            llm_cfg=llm_cfg,
            max_steps_per_operator=max_steps_per_operator,
            selective_fallback_rule=selective_fallback_rule,
            use_optimized_agent=use_optimized_agent,
            max_time_per_operator=max_time_per_operator,
            max_index_validation_retries=max_index_validation_retries,
            dependency_threshold=dependency_threshold,
            enable_rollback=enable_rollback,
        )

    @staticmethod
    def scan_workspace(
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        max_files: int = 300,
        max_files_per_dir: int = 50,
        timeout: float = 60.0,
    ) -> str:
        from src.agent.code_agent_plan_mode import CodeAgentPlanMode
        return CodeAgentPlanMode.scan_workspace(
            workspace=workspace,
            repo_path=repo_path,
            max_files=max_files,
            max_files_per_dir=max_files_per_dir,
            timeout=timeout,
        )

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> DAGPipelineResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_dag")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info(f"[DAGPlanningExecution] Scanning workspace at {repo_path}...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[DAGPlanningExecution] Workspace scan complete: {len(workspace_overview)} chars")
        self._write_text(os.path.join(logs_dir, "workspace_overview.txt"), workspace_overview)

        logger.info("[DAGPlanningExecution] Generating DAG operator plan...")
        plan, planning_metrics = self.planning_agent.generate_plan(
            instruction=instruction,
            workspace_overview=workspace_overview,
            output_dir=logs_dir,
        )
        self._write_json(os.path.join(logs_dir, "plan.json"), plan.to_dict_list())
        logger.info(f"[DAGPlanningExecution] Plan: {len(plan.operators)} operators")
        for op in plan.operators:
            deps_str = ", ".join(f"Op {k}({v:.1f})" for k, v in sorted(op.dependencies.items()))
            logger.info(f"  Op {op.index}: {op.subtask[:80]} [deps: {deps_str}]")

        logger.info("[DAGPlanningExecution] Building dependency graph...")
        graph = build_dependency_graph(
            plan=plan,
            dependency_threshold=self.dependency_threshold,
        )
        self._write_json(os.path.join(logs_dir, "dependency_graph.json"), graph.to_dict())

        logger.info("[DAGPlanningExecution] Executing DAG operator plan...")
        exec_results, exec_total_metrics = self.executor.execute_plan(
            plan=plan,
            graph=graph,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            callbacks=callbacks,
            output_dir=logs_dir,
            max_rollback_attempts=self.max_rollback_attempts,
        )

        exec_total = exec_total_metrics["total"]
        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "total": _merge_metrics_dicts(planning_metrics, exec_total),
        }
        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)

        results_payload = {
            "plan": plan.to_dict_list(),
            "graph": graph.to_dict(),
            "exec_results": [
                {
                    "operator_index": r.operator_index,
                    "operator_subtask": r.operator_subtask,
                    "finish_message": r.finish_message,
                    "metrics": r.metrics,
                    "error": r.error,
                    "files_modified": r.files_modified,
                    "files_read": r.files_read,
                    "hit_max_steps": r.hit_max_steps,
                    "useful_trajectory_indexes": r.useful_trajectory_indexes,
                    "messages_count": len(r.messages),
                    "trajectory_records_count": len(r.trajectory_records),
                }
                for r in exec_results
            ],
            "metrics_summary": total_metrics,
        }
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        other = {
            "logs_dir": logs_dir,
            "plan": plan.to_dict_list(),
            "graph": graph.to_dict(),
        }

        return DAGPipelineResult(
            metrics=total_metrics,
            plan=plan,
            graph=graph,
            exec_results=exec_results,
            other_content=other,
        )

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
