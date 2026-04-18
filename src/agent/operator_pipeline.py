from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace
    from src.agent.code_agent import AgentResult

from src.agent.operator import Operator, OperatorPlan, OperatorCEResult, LLMBackbone
from src.agent.operator_planning_agent import OperatorPlanningAgent
from src.agent.operator_ce_agent import OperatorCEAgent
from src.agent.operator_rewriter import OperatorRewriter
from src.agent.operator_execution_agent import OperatorExecutionAgent, OperatorExecResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None
    plan: OperatorPlan | None = None
    rewritten_plan: OperatorPlan | None = None
    ce_results: list[OperatorCEResult] = field(default_factory=list)
    exec_results: list[OperatorExecResult] = field(default_factory=list)
    rewrite_actions: list[dict[str, Any]] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


class OperatorPipeline:
    def __init__(
        self,
        planner_cfg: dict,
        ce_cfg: dict,
        rewrite_cfg: dict | None = None,
        llm_configs: dict[str, dict] | None = None,
        config_dir: str = "./_config",
        use_ce: bool = True,
        use_rewrite: bool = True,
        use_rule_rewrite: bool = True,
        use_llm_rewrite: bool = True,
        max_steps_per_operator: int = 30,
        ce_total_timeout: int = 300,
        ce_operator_timeout: int = 60,
        ce_max_attempts: int = 1,
        max_rewrite_rounds: int = 1,
        trajectory_passing_mode: str = "trajectory",
    ):
        self.config_dir = config_dir
        self.use_ce = use_ce
        self.use_rewrite = use_rewrite
        self.use_rule_rewrite = use_rule_rewrite
        self.use_llm_rewrite = use_llm_rewrite
        self.ce_total_timeout = ce_total_timeout
        self.ce_operator_timeout = ce_operator_timeout
        self.ce_max_attempts = ce_max_attempts
        self.max_rewrite_rounds = max_rewrite_rounds

        if llm_configs is None:
            llm_configs = self._load_llm_configs(config_dir)
        self.llm_configs = llm_configs

        self.planning_agent = OperatorPlanningAgent(planner_cfg=planner_cfg)

        self.ce_agent = OperatorCEAgent(
            ce_cfg=ce_cfg,
            llm_configs=llm_configs,
        ) if use_ce else None

        self.rewriter = OperatorRewriter(
            rewrite_cfg=rewrite_cfg,
            use_rule_based=use_rule_rewrite,
            use_llm=use_llm_rewrite,
        ) if use_rewrite else None

        self.execution_agent = OperatorExecutionAgent(
            llm_configs=llm_configs,
            max_steps_per_operator=max_steps_per_operator,
            config_dir=config_dir,
            trajectory_passing_mode=trajectory_passing_mode,
        )

    @staticmethod
    def _load_llm_configs(config_dir: str) -> dict[str, dict]:
        configs = {}
        mapping = {
            "A": "doubao_flash.yaml",
            "B": "doubao.yaml",
            "C": "kimi2.5.yaml",
        }
        for key, filename in mapping.items():
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                cfg = yaml.safe_load(open(path, "r"))
                configs[key] = cfg
            else:
                logger.warning(f"Config file not found: {path}")
        return configs

    @staticmethod
    def scan_workspace(
        workspace: DockerWorkspace,
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
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PipelineResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        # ---- Phase 0: Workspace Scan ----
        logger.info(f"[OperatorPipeline] Scanning workspace at {repo_path}...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[OperatorPipeline] Workspace scan complete: {len(workspace_overview)} chars")
        self._write_text(os.path.join(logs_dir, "workspace_overview.txt"), workspace_overview)

        # ---- Phase 1: Planning ----
        logger.info("[OperatorPipeline] Phase 1: Generating operator plan...")
        plan = self.planning_agent.generate_plan(
            instruction=instruction,
            workspace_overview=workspace_overview,
        )
        self._write_json(os.path.join(logs_dir, "initial_plan.json"), plan.to_dict_list())
        logger.info(f"[OperatorPipeline] Initial plan: {len(plan.operators)} operators")
        for op in plan.operators:
            logger.info(f"  Op {op.index}: [{op.llm_backbone.value}] {op.subtask[:80]}")

        # ---- Phase 2: Cost Estimation ----
        ce_results: list[OperatorCEResult] = []
        ce_metrics: dict[str, Any] = {}
        if self.use_ce and self.ce_agent:
            logger.info("[OperatorPipeline] Phase 2: Estimating costs...")
            ce_results, ce_metrics = self.ce_agent.estimate_plan(
                instruction=instruction,
                plan=plan,
                total_timeout=self.ce_total_timeout,
                operator_timeout=self.ce_operator_timeout,
                max_attempts=self.ce_max_attempts,
            )
            self._write_json(
                os.path.join(logs_dir, "ce_results.json"),
                [r.to_dict() for r in ce_results],
            )

            initial_plan_with_ce = []
            for op in plan.operators:
                op_dict = op.to_dict()
                ce_for_op = next((r for r in ce_results if r.operator_index == op.index), None)
                if ce_for_op and not ce_for_op.error:
                    op_dict["ce_uncertainty"] = ce_for_op.uncertainty
                    op_dict["ce_estimated_cost"] = ce_for_op.estimated_cost
                    op_dict["ce_predicted_steps"] = ce_for_op.predicted_steps
                initial_plan_with_ce.append(op_dict)
            self._write_json(
                os.path.join(logs_dir, "initial_plan_with_ce.json"),
                initial_plan_with_ce,
            )
            for r in ce_results:
                if not r.error:
                    logger.info(
                        f"  Op {r.operator_index}: cost={r.estimated_cost:.6f}, "
                        f"uncertainty={r.uncertainty:.2f}, steps={r.predicted_steps}"
                    )

        # ---- Phase 3: Rewriting ----
        rewritten_plan = plan
        rewrite_actions: list[dict[str, Any]] = []
        if self.use_rewrite and self.rewriter and ce_results:
            logger.info("[OperatorPipeline] Phase 3: Rewriting plan...")
            for round_idx in range(self.max_rewrite_rounds):
                current_ce = ce_results if round_idx == 0 else new_ce_results

                rewritten_plan, rewrite_actions = self.rewriter.rewrite(
                    instruction=instruction,
                    plan=rewritten_plan,
                    ce_results=current_ce,
                )

                if not rewrite_actions:
                    logger.info(f"[OperatorPipeline] No rewrite actions in round {round_idx + 1}, stopping")
                    break

                self._write_json(
                    os.path.join(logs_dir, f"rewritten_plan_round_{round_idx + 1}.json"),
                    rewritten_plan.to_dict_list(),
                )
                self._write_json(
                    os.path.join(logs_dir, f"rewrite_actions_round_{round_idx + 1}.json"),
                    rewrite_actions,
                )

                if self.use_ce and self.ce_agent and round_idx < self.max_rewrite_rounds - 1:
                    logger.info(f"[OperatorPipeline] Re-estimating costs for rewritten plan (round {round_idx + 1})...")
                    new_ce_results, new_ce_metrics = self.ce_agent.estimate_plan(
                        instruction=instruction,
                        plan=rewritten_plan,
                        total_timeout=self.ce_total_timeout,
                        operator_timeout=self.ce_operator_timeout,
                        max_attempts=self.ce_max_attempts,
                    )
                    ce_metrics = new_ce_metrics
                else:
                    break

        self._write_json(os.path.join(logs_dir, "final_plan.json"), rewritten_plan.to_dict_list())
        logger.info(f"[OperatorPipeline] Final plan: {len(rewritten_plan.operators)} operators")
        for op in rewritten_plan.operators:
            logger.info(f"  Op {op.index}: [{op.llm_backbone.value}] {op.subtask[:80]}")

        # ---- Phase 4: Execution ----
        logger.info("[OperatorPipeline] Phase 4: Executing operator plan...")
        exec_results = self.execution_agent.execute_plan(
            plan=rewritten_plan,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            callbacks=callbacks,
            output_dir=logs_dir,
        )

        # ---- Phase 5: Summary ----
        exec_total_metrics = OperatorExecutionAgent.compute_total_metrics(exec_results)

        total_metrics = {
            "planning": {},
            "cost_estimation": ce_metrics,
            "execution": exec_total_metrics["total"],
            "execution_by_backbone": exec_total_metrics["by_backbone"],
            "total": {
                "prompt_tokens": ce_metrics.get("prompt_tokens", 0) + exec_total_metrics["total"].get("prompt_tokens", 0),
                "completion_tokens": ce_metrics.get("completion_tokens", 0) + exec_total_metrics["total"].get("completion_tokens", 0),
                "cache_read_tokens": ce_metrics.get("cache_read_tokens", 0) + exec_total_metrics["total"].get("cache_read_tokens", 0),
                "reasoning_tokens": ce_metrics.get("reasoning_tokens", 0) + exec_total_metrics["total"].get("reasoning_tokens", 0),
                "total_tokens": ce_metrics.get("total_tokens", 0) + exec_total_metrics["total"].get("total_tokens", 0),
                "accumulated_cost": ce_metrics.get("accumulated_cost", 0.0) + exec_total_metrics["total"].get("accumulated_cost", 0.0),
            },
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)

        results_payload = {
            "initial_plan": plan.to_dict_list(),
            "final_plan": rewritten_plan.to_dict_list(),
            "ce_results": [r.to_dict() for r in ce_results],
            "rewrite_actions": rewrite_actions,
            "exec_results": [
                {
                    "operator_index": r.operator_index,
                    "llm_backbone": r.llm_backbone,
                    "finish_message": r.finish_message[:500],
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in exec_results
            ],
            "metrics_summary": total_metrics,
        }
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [
            "# OperatorPipeline Run Log",
            "",
            f"- Timestamp: {now}",
            f"- Initial plan operators: {len(plan.operators)}",
            f"- Final plan operators: {len(rewritten_plan.operators)}",
            f"- CE enabled: {self.use_ce}",
            f"- Rewrite enabled: {self.use_rewrite}",
            f"- Rewrite actions: {len(rewrite_actions)}",
            f"- Logs Dir: `{logs_dir}`",
            "",
            "## Final Plan",
        ]
        for op in rewritten_plan.operators:
            log_lines.append(f"- Op {op.index}: [{op.llm_backbone.value}] {op.subtask}")
        log_lines.extend([
            "",
            "## Execution Summary",
        ])
        for r in exec_results:
            status = "OK" if r.error is None else f"ERROR: {r.error}"
            log_lines.append(
                f"- Op {r.operator_index} [{r.llm_backbone}]: {status} "
                f"(tokens: {r.metrics.get('total_tokens', 0)})"
            )
        self._write_text(os.path.join(logs_dir, "log.md"), "\n".join(log_lines))

        other = {
            "logs_dir": logs_dir,
            "initial_plan": plan.to_dict_list(),
            "final_plan": rewritten_plan.to_dict_list(),
            "ce_results": [r.to_dict() for r in ce_results],
            "rewrite_actions": rewrite_actions,
        }

        return PipelineResult(
            metrics=total_metrics,
            plan=plan,
            rewritten_plan=rewritten_plan,
            ce_results=ce_results,
            exec_results=exec_results,
            rewrite_actions=rewrite_actions,
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
