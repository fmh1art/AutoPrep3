"""
DA-Code benchmark integration for AutoPrep3.

DA-Code: Agent Data Science Code Generation Benchmark
Repository: https://github.com/yiyihum/da-code
Paper: EMNLP 2024

This package provides:
- lib_tasks: Task data model + JSONL config loader
- lib_grading: Deterministic grading (CSV/text/image/ML metric comparison)
- dacode_runner: Main runner (workspace prep + agent execution + grading)
"""

from .lib_tasks import Task, TaskLoader
from .dacode_runner import DACodeRunner

__all__ = ["Task", "TaskLoader", "DACodeRunner"]
