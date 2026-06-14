"""
DA-Code task loader - parses JSONL config files from the DA-Code benchmark.

Original repo: https://github.com/yiyihum/da-code

Data layout on disk:
    <dataset_root>/
        configs/
            task/
                examples.jsonl    # one JSON object per line: {id, instruction, evaluator, config, post_process}
            eval/
                all.jsonl         # one JSON object per line: {id, func, conj, result, options, config}
        source/
            <task_id>/           # input data files (CSV/JSON/Excel/SQL)
        gold/
            <task_id>/           # gold-standard output files for grading

Task config fields (from examples.jsonl):
    id: str              - unique task id
    instruction: str     - natural-language task prompt for the agent
    evaluator: dict      - {func, conj, result, options}  (may be list)
    post_process: list   - optional post-processing steps (e.g., ["plot_process"])
    config: dict         - {task: category, hardness, type}

Eval config fields (from all.jsonl):
    id: str              - must match task config id
    func: str | list[str] - metric function name(s)
    conj: str            - "avg" | "max" | "min" | "and" | "or"
    result: dict | list[dict] - {file: "output.csv"} or {number: 42.5} or similar
    options: dict | list[dict] - options passed to each metric
    config: dict         - metadata

Note: the evaluator dict in the task config is a fallback — the eval config
(in configs/eval/all.jsonl) has the authoritative grading specification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricSpec:
    """Specification for a single grading metric."""
    func: str
    result: Dict[str, Any]  # e.g. {"file": "output.csv"} or {"number": 42.5}
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def result_type(self) -> str:
        """Return the kind of result: 'file', 'number', or 'unknown'."""
        if "file" in self.result:
            return "file"
        if "number" in self.result:
            return "number"
        return "unknown"

    @property
    def result_value(self) -> Any:
        """Return the raw result value (file path or number)."""
        if self.result_type == "file":
            return self.result["file"]
        if self.result_type == "number":
            return self.result["number"]
        return None


@dataclass
class Task:
    """A DA-Code benchmark task."""
    task_id: str
    instruction: str
    category: str
    hardness: str
    metric_funcs: List[MetricSpec]
    conjunction: str  # "avg" | "max" | "min" | "and" | "or"
    post_process: List[str] = field(default_factory=list)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "category": self.category,
            "hardness": self.hardness,
            "metric_funcs": [
                {"func": m.func, "result": m.result, "options": m.options}
                for m in self.metric_funcs
            ],
            "conjunction": self.conjunction,
            "post_process": self.post_process,
        }

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id!r}, category={self.category!r}, "
            f"hardness={self.hardness!r}, n_metrics={len(self.metric_funcs)})"
        )


class TaskLoader:
    """Loads DA-Code tasks from a dataset directory."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"DA-Code dataset root not found: {self.dataset_root}")
        logger.info("DA-Code dataset root: %s", self.dataset_root)

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load all JSON objects from a JSONL file."""
        if not path.exists():
            logger.warning("JSONL config not found: %s", path)
            return []
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Malformed JSON at %s line %d: %s (skipping)",
                        path, line_num, e,
                    )
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    def _find_config(self, subdir: str, basename: str) -> Optional[Path]:
        """Locate a config file, trying multiple candidate paths."""
        candidates = [
            self.dataset_root / subdir / basename,
            self.dataset_root / f"{subdir}/{basename}",
            self.dataset_root / "configs" / subdir / basename,
            self.dataset_root / basename,
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def load_all_tasks(self) -> List[Task]:
        """Load all tasks from the dataset (matching task + eval configs).

        Priority: configs/eval/*.jsonl provides the authoritative grading
        spec. configs/task/*.jsonl provides the task prompt (instruction).
        """
        # --- Load eval configs (authoritative list of tasks) ---
        eval_path = self._find_config("eval", "all.jsonl")
        if eval_path is None:
            eval_path = self._find_config("eval", "eval.jsonl")
        if eval_path is None:
            # Fall back to scanning any jsonl in configs/
            for p in sorted(self.dataset_root.rglob("*.jsonl")):
                if "task" in p.name.lower():
                    eval_path = p
                    break
        eval_records = self._load_jsonl(eval_path) if eval_path is not None else []

        # --- Load task configs (instruction prompts) ---
        task_path = self._find_config("task", "examples.jsonl")
        if task_path is None:
            task_path = self._find_config("task", "train.jsonl")
        task_records = self._load_jsonl(task_path) if task_path is not None else []

        # --- Index prompts by id ---
        prompts_by_id: Dict[str, str] = {}
        for rec in task_records:
            tid = rec.get("id")
            if tid and "instruction" in rec:
                prompts_by_id[str(tid)] = rec["instruction"]

        # --- Build Task objects ---
        source_dir = self.dataset_root / "source"
        tasks: List[Task] = []
        for rec in eval_records if eval_records else task_records:
            task_id = str(rec.get("id", ""))
            if not task_id:
                continue

            # instruction: prefer task config; fall back to eval config 'instruction'
            instruction = prompts_by_id.get(task_id, rec.get("instruction", ""))
            if not instruction:
                logger.warning("Task %s has no instruction; skipping", task_id)
                continue

            # metric functions
            raw_func = rec.get("func")
            raw_result = rec.get("result", {})
            raw_options = rec.get("options", {})
            raw_conj = rec.get("conj", "avg")

            if isinstance(raw_func, list):
                funcs = raw_func
            else:
                funcs = [raw_func] if raw_func else ["compare_csv"]

            # Normalize result to list matching funcs
            if isinstance(raw_result, list):
                results = raw_result
            else:
                results = [raw_result] * len(funcs)

            # Normalize options to list matching funcs
            if isinstance(raw_options, list):
                options_list = raw_options
            else:
                options_list = [raw_options or {}] * len(funcs)

            metric_funcs: List[MetricSpec] = []
            for func_name, res, opts in zip(funcs, results, options_list):
                if not isinstance(func_name, str):
                    func_name = str(func_name)
                if res is None:
                    res = {}
                if opts is None:
                    opts = {}
                metric_funcs.append(
                    MetricSpec(func=func_name, result=res, options=opts)
                )

            # config.metadata
            cfg = rec.get("config") or {}
            if not isinstance(cfg, dict):
                cfg = {}
            category = cfg.get("task", rec.get("task_type", "data_insight"))
            hardness = cfg.get("hardness", rec.get("hardness", "easy"))

            # post_process
            post_process = rec.get("post_process", [])
            if isinstance(post_process, str):
                post_process = [post_process]

            task = Task(
                task_id=task_id,
                instruction=instruction,
                category=str(category),
                hardness=str(hardness),
                metric_funcs=metric_funcs,
                conjunction=str(raw_conj),
                post_process=list(post_process),
                raw_config=rec,
            )
            tasks.append(task)

        # Filter out tasks for which we have a source directory (when available)
        if source_dir.exists():
            has_source = {p.name for p in source_dir.iterdir() if p.is_dir()}
            n_before = len(tasks)
            tasks = [t for t in tasks if t.task_id in has_source]
            logger.info(
                "Kept %d / %d tasks with source/ directories", len(tasks), n_before,
            )
        else:
            logger.warning(
                "No source/ directory at %s; proceeding with %d raw tasks",
                source_dir, len(tasks),
            )

        logger.info("Loaded %d DA-Code tasks", len(tasks))
        return tasks

    def get_task_ids_by_category(self, category: str) -> List[str]:
        """Return task ids matching a given category (case-insensitive)."""
        all_tasks = self.load_all_tasks()
        cat_lower = category.lower()
        return [t.task_id for t in all_tasks if t.category.lower() == cat_lower]

    def get_task_ids_by_hardness(self, hardness: str) -> List[str]:
        """Return task ids matching a hardness level."""
        all_tasks = self.load_all_tasks()
        h_lower = hardness.lower()
        return [t.task_id for t in all_tasks if t.hardness.lower() == h_lower]
