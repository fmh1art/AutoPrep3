"""
Draw the full DAG and the minimum DAG sub-graph for each trajectory.

The minimum DAG (mini DAG) is extracted by reverse-BFS from the last node,
keeping only the steps that are actually needed to reach the final result.
This is the same algorithm used in SkillEvolver.extract_mini_dag().

Usage:
    python example/pre_exp/draw_mini_dag.py --root _tmp/code_agent_limit64_doubao_2026-05-16_18-59-41
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import re

from src.tools.funcs import all_filepaths_in_dir, open_json


def _parse_meta_string(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(s.lstrip())
        return obj


# ---------------------------------------------------------------------------
# DAG data structures (mirrored from src/agent/skill_evolver.py)
# ---------------------------------------------------------------------------

class DAGNode:
    __slots__ = ("index", "meta_data", "in_nodes", "out_nodes")

    def __init__(self, index: int, meta_data: dict, in_nodes: list | None = None, out_nodes: list | None = None):
        self.index = index
        self.meta_data = meta_data
        self.in_nodes: list["DAGNode"] = in_nodes or []
        self.out_nodes: list["DAGNode"] = out_nodes or []


class DAG:
    def __init__(self):
        self.nodes: dict[int, DAGNode] = {}

    @classmethod
    def build_from_step2meta_info(cls, step2meta_info: dict) -> "DAG":
        dag = cls()
        for step_idx_str, meta in step2meta_info.items():
            step_idx = int(step_idx_str)
            if isinstance(meta, str):
                meta = _parse_meta_string(meta)
            dag.nodes[step_idx] = DAGNode(index=step_idx, meta_data=meta)
        for step_idx, node in dag.nodes.items():
            for dep_idx in node.meta_data.get("related_step_indexs", []):
                dep_node = dag.nodes.get(dep_idx)
                if dep_node is not None:
                    node.in_nodes.append(dep_node)
                    dep_node.out_nodes.append(node)
        return dag

    def get_tol_step_idxs(self) -> list[int]:
        return list(self.nodes.keys())


def extract_mini_dag(dag: DAG) -> DAG:
    if not dag.nodes:
        return DAG()
    last_idx = max(dag.nodes.keys())
    visited = set()
    stack = [dag.nodes[last_idx]]
    while stack:
        node = stack.pop()
        if node.index in visited:
            continue
        visited.add(node.index)
        for in_node in node.in_nodes:
            if in_node.index not in visited:
                stack.append(in_node)
    mini = DAG()
    for idx in visited:
        mini.nodes[idx] = DAGNode(index=idx, meta_data=dag.nodes[idx].meta_data)
    for idx, node in mini.nodes.items():
        for dep_idx in node.meta_data.get("related_step_indexs", []):
            dep_node = mini.nodes.get(dep_idx)
            if dep_node is not None:
                node.in_nodes.append(dep_node)
                dep_node.out_nodes.append(node)
    return mini


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

INTENTION_COLORS = {
    "Progressive": "#4CAF50",
    "Refinement": "#FF9800",
}
EFECTION_SHAPES = {
    "Read": "ellipse",
    "Write": "box",
    "Meta": "diamond",
}
REMOVED_NODE_COLOR = "#B0BEC5"
MINI_EDGE_COLOR = "#1565C0"
REMOVED_EDGE_COLOR = "#90A4AE"


def _node_label(node: DAGNode) -> str:
    intention = node.meta_data.get("intention", "?")
    effection = node.meta_data.get("effection", "?")
    return f"Step {node.index}\n{intention}/{effection}"


def draw_full_dag(dag: DAG, save_path: str):
    import graphviz as gv

    dot = gv.Digraph(name="full_dag", format="png")
    dot.attr(rankdir="TB", dpi="150")
    dot.attr("node", fontname="sans-serif", fontsize="10", style="filled")
    dot.attr("edge", fontname="sans-serif", fontsize="8")

    for idx in sorted(dag.nodes.keys()):
        node = dag.nodes[idx]
        intention = node.meta_data.get("intention", "?")
        effection = node.meta_data.get("effection", "?")
        fillcolor = INTENTION_COLORS.get(intention, "#E0E0E0")
        shape = EFECTION_SHAPES.get(effection, "ellipse")
        dot.node(
            name=str(idx),
            label=_node_label(node),
            shape=shape,
            fillcolor=fillcolor,
            fontcolor="white",
        )

    for idx in sorted(dag.nodes.keys()):
        node = dag.nodes[idx]
        for in_node in node.in_nodes:
            dot.edge(str(in_node.index), str(idx))

    dot.render(filename=save_path, cleanup=True)


def draw_mini_dag(full_dag: DAG, mini_dag: DAG, save_path: str):
    import graphviz as gv

    mini_idxs = set(mini_dag.nodes.keys())

    dot = gv.Digraph(name="mini_dag", format="png")
    dot.attr(rankdir="TB", dpi="150")
    dot.attr("node", fontname="sans-serif", fontsize="10", style="filled")
    dot.attr("edge", fontname="sans-serif", fontsize="8")

    for idx in sorted(full_dag.nodes.keys()):
        node = full_dag.nodes[idx]
        if idx in mini_idxs:
            intention = node.meta_data.get("intention", "?")
            effection = node.meta_data.get("effection", "?")
            fillcolor = INTENTION_COLORS.get(intention, "#E0E0E0")
            shape = EFECTION_SHAPES.get(effection, "ellipse")
            dot.node(
                name=str(idx),
                label=_node_label(node),
                shape=shape,
                fillcolor=fillcolor,
                fontcolor="white",
                penwidth="2",
            )
        else:
            effection = node.meta_data.get("effection", "?")
            shape = EFECTION_SHAPES.get(effection, "ellipse")
            dot.node(
                name=str(idx),
                label=_node_label(node),
                shape=shape,
                fillcolor=REMOVED_NODE_COLOR,
                fontcolor="#546E7A",
                style="filled,dashed",
            )

    mini_edges = set()
    for idx in sorted(mini_dag.nodes.keys()):
        node = mini_dag.nodes[idx]
        for in_node in node.in_nodes:
            mini_edges.add((in_node.index, idx))

    for idx in sorted(full_dag.nodes.keys()):
        node = full_dag.nodes[idx]
        for in_node in node.in_nodes:
            edge_key = (in_node.index, idx)
            if edge_key in mini_edges:
                dot.edge(
                    str(in_node.index), str(idx),
                    color=MINI_EDGE_COLOR,
                    penwidth="1.5",
                )
            else:
                dot.edge(
                    str(in_node.index), str(idx),
                    color=REMOVED_EDGE_COLOR,
                    style="dashed",
                    penwidth="0.8",
                )

    dot.render(filename=save_path, cleanup=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(root: str):
    data = []
    log_dir = os.path.join(root, "log")
    if not os.path.isdir(log_dir):
        print(f"[Warning] log directory not found: {log_dir}")
        return data

    for case_id in sorted(os.listdir(log_dir)):
        case_dir = os.path.join(log_dir, case_id)
        if not os.path.isdir(case_dir):
            continue

        step2meta_path = os.path.join(case_dir, "step2meta_info.json")
        if not os.path.isfile(step2meta_path):
            continue

        step2meta = open_json(step2meta_path)
        if not step2meta:
            continue

        data.append((case_id, case_dir, step2meta))

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Draw full DAG and mini DAG for each trajectory.")
    parser.add_argument("--root", type=str,
                        default="_tmp/code_agent_limit64_doubao_2026-05-16_18-59-41",
                        help="Experiment output directory containing log/ subdirectory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.root)
    print(f"Found {len(data)} instances with step2meta_info")

    for case_id, case_dir, step2meta in data:
        full_dag = DAG.build_from_step2meta_info(step2meta)
        mini_dag = extract_mini_dag(full_dag)

        full_count = len(full_dag.nodes)
        mini_count = len(mini_dag.nodes)
        print(f"[{case_id}] full DAG: {full_count} steps, mini DAG: {mini_count} steps "
              f"({mini_count / max(full_count, 1):.1%} retained)")

        full_dag_path = os.path.join(case_dir, "dag_full")
        mini_dag_path = os.path.join(case_dir, "dag_mini")

        draw_full_dag(full_dag, full_dag_path)
        draw_mini_dag(full_dag, mini_dag, mini_dag_path)

        print(f"  -> saved {full_dag_path}.png and {mini_dag_path}.png")

    print(f"\nDone. Processed {len(data)} instances.")
