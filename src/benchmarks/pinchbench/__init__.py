"""
PinchBench (skill benchmark) integration for AutoPrep3.

This module provides integration with the PinchBench benchmark
(https://github.com/pinchbench/skill.git) for evaluating code agents
on real-world tasks.
"""

from .lib_tasks import Task, TaskLoader
from .pinchbench_runner import PinchBenchRunner

__all__ = ["Task", "TaskLoader", "PinchBenchRunner"]
