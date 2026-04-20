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

import copy

from src.agent.operator import Operator, OperatorPlan, OperatorCEResult, LLMBackbone, CognitiveType, set_backbone_display_names
from src.agent.operator_planning_agent import OperatorPlanningAgent
from src.agent.operator_ce_agent import OperatorCEAgent
from src.agent.operator_rewriter import OperatorRewriter
from src.agent.operator_execution_agent import OperatorExecutionAgent, OperatorExecResult
from src.agent.backbone_selector import BackboneSelector

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
        trajectory_passing_mode: str = "hybrid",
        fallback_upgrade: bool = True,
        selective_max_retries: int = 2,
        selective_fallback_rule: str = "last_half",
        force_backbone: str | None = None,
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
        self.fallback_upgrade = fallback_upgrade
        self.force_backbone = force_backbone

        if llm_configs is None:
            llm_configs = self._load_llm_configs(config_dir)
        else:
            display_names = {}
            for key, cfg in llm_configs.items():
                display_names[key] = cfg.get("llm_name", key)
            set_backbone_display_names(display_names)
        self.llm_configs = llm_configs

        self.planning_agent = OperatorPlanningAgent(planner_cfg=planner_cfg)

        self.ce_agent = OperatorCEAgent(
            ce_cfg=ce_cfg,
            llm_configs=llm_configs,
        ) if use_ce else None

        self.backbone_selector = BackboneSelector(
            ce_agent=self.ce_agent,
        ) if use_ce and self.ce_agent else None

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
            fallback_upgrade=fallback_upgrade,
            selective_max_retries=selective_max_retries,
            selective_fallback_rule=selective_fallback_rule,
        )

    @staticmethod
    def _load_llm_configs(config_dir: str) -> dict[str, dict]:
        configs = {}
        mapping = {
            "CHEAP": "doubao.yaml",
            "EXPENSIVE": "kimi2.5.yaml",
        }
        display_names = {}
        for key, filename in mapping.items():
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                cfg = yaml.safe_load(open(path, "r"))
                configs[key] = cfg
                display_names[key] = cfg.get("llm_name", key)
            else:
                logger.warning(f"Config file not found: {path}")
        set_backbone_display_names(display_names)
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
        plan, planning_metrics = self.planning_agent.generate_plan(
            instruction=instruction,
            workspace_overview=workspace_overview,
            workspace=workspace,
            repo_path=repo_path,
        )
        self._write_json(os.path.join(logs_dir, "initial_plan.json"), plan.to_dict_list())
        logger.info(f"[OperatorPipeline] Initial plan: {len(plan.operators)} operators")
        for op in plan.operators:
            bb_str = f"[{op.llm_backbone.value}]" if op.llm_backbone else "[unassigned]"
            logger.info(f"  Op {op.index}: {bb_str} {op.subtask[:80]}")

        # ---- Phase 1.5: Information Closure Validation ----
        plan = self._validate_information_closure(plan)
        self._write_json(os.path.join(logs_dir, "plan_after_info_closure.json"), plan.to_dict_list())

        # ---- Phase 1.6: Backbone Assignment by Cognitive Type ----
        if self.force_backbone:
            forced_bb = LLMBackbone.from_str(self.force_backbone)
            logger.info(
                f"[OperatorPipeline] Force backbone: overriding all operators to {forced_bb.value}"
            )
            for op in plan.operators:
                op.llm_backbone = forced_bb
        else:
            for op in plan.operators:
                if op.cognitive_type is not None:
                    op.llm_backbone = op.cognitive_type.default_backbone
                elif op.llm_backbone is None:
                    op.llm_backbone = LLMBackbone.CHEAP
            logger.info("[OperatorPipeline] Backbones assigned by cognitive type")
            for op in plan.operators:
                cog = op.cognitive_type.value if op.cognitive_type else "none"
                logger.info(f"  Op {op.index}: [{cog}] → {op.llm_backbone.value} | {op.subtask[:60]}")

        # ---- Phase 2+3: Iterative CE + Rewrite ----
        ce_results: list[OperatorCEResult] = []
        ce_metrics: dict[str, Any] = {}
        selection_details: dict[int, dict[str, Any]] = {}
        rewrite_actions: list[dict[str, Any]] = []
        rewrite_metrics: dict[str, Any] = {}
        rewritten_plan = plan

        if self.use_ce and self.ce_agent:
            logger.info("[OperatorPipeline] Phase 2: CE-based cost & uncertainty estimation...")

            ce_matrix, batch_metrics = self.ce_agent.estimate_plan_batch(
                instruction=instruction,
                plan=rewritten_plan,
                candidate_backbones=LLMBackbone.available_backbones(),
            )

            usage_after_ce = self.ce_agent.caller.get_total_usage()
            ce_metrics = {
                "prompt_tokens": usage_after_ce.get("input_tokens", 0),
                "completion_tokens": usage_after_ce.get("output_tokens", 0),
                "cache_read_tokens": usage_after_ce.get("cached_tokens", 0),
                "reasoning_tokens": usage_after_ce.get("reasoning_tokens", 0),
                "total_tokens": usage_after_ce.get("total_tokens", 0),
                "accumulated_cost": 0.0,
                "llm_backbone": self.ce_agent._get_llm_backbone(),
            }

            has_cognitive_types = any(op.cognitive_type is not None for op in rewritten_plan.operators)

            if has_cognitive_types and self.backbone_selector and not self.force_backbone:
                logger.info("[OperatorPipeline] Using cognitive-type-based backbone selection...")
                rewritten_plan, selection_details, ce_results = self.backbone_selector.select_by_cognitive_type(
                    instruction=instruction,
                    plan=rewritten_plan,
                )

            for op in rewritten_plan.operators:
                cheap_ce = ce_matrix.get(op.index, {}).get("CHEAP")
                exp_ce = ce_matrix.get(op.index, {}).get("EXPENSIVE")
                cheap_u = cheap_ce.uncertainty if cheap_ce and not cheap_ce.error else 0.5
                exp_u = exp_ce.uncertainty if exp_ce and not exp_ce.error else 0.5
                cheap_cost = cheap_ce.estimated_cost if cheap_ce and not cheap_ce.error else 0.0
                exp_cost = exp_ce.estimated_cost if exp_ce and not exp_ce.error else 0.0
                logger.info(
                    f"  Op {op.index} [{op.llm_backbone.value}]: "
                    f"CHEAP u={cheap_u:.2f} cost=${cheap_cost:.6f} | "
                    f"EXPENSIVE u={exp_u:.2f} cost=${exp_cost:.6f}"
                )

            flat_ce = []
            for op in rewritten_plan.operators:
                ce_result = ce_matrix.get(op.index, {}).get(op.llm_backbone.value)
                if ce_result:
                    flat_ce.append(ce_result)
            ce_results = flat_ce

            self._write_json(
                os.path.join(logs_dir, "ce_results.json"),
                [r.to_dict() for r in ce_results],
            )

            if self.use_rewrite and self.rewriter:
                logger.info("[OperatorPipeline] Phase 3: Iterative rewrite...")

                for round_idx in range(self.max_rewrite_rounds):
                    logger.info(f"  Rewrite round {round_idx + 1}/{self.max_rewrite_rounds}")

                    current_ce_for_rewrite = []
                    for op in rewritten_plan.operators:
                        ce_result = ce_matrix.get(op.index, {}).get(op.llm_backbone.value)
                        if ce_result:
                            current_ce_for_rewrite.append(ce_result)
                        else:
                            current_ce_for_rewrite.append(OperatorCEResult(
                                operator_index=op.index, uncertainty=0.5, error=True,
                            ))

                    rewritten_plan, round_actions, round_rewrite_metrics = self.rewriter.rewrite(
                        instruction=instruction,
                        plan=rewritten_plan,
                        ce_results=current_ce_for_rewrite,
                        ce_matrix=ce_matrix,
                    )
                    rewrite_metrics = round_rewrite_metrics
                    rewrite_actions.extend(round_actions)

                    if not round_actions:
                        logger.info(f"  No rewrite actions in round {round_idx + 1}, stopping")
                        break

                    self._write_json(
                        os.path.join(logs_dir, f"rewritten_plan_round_{round_idx + 1}.json"),
                        rewritten_plan.to_dict_list(),
                    )
                    self._write_json(
                        os.path.join(logs_dir, f"rewrite_actions_round_{round_idx + 1}.json"),
                        round_actions,
                    )

                    if round_idx < self.max_rewrite_rounds - 1:
                        logger.info(f"  Re-estimating costs for rewritten plan (round {round_idx + 1})...")
                        try:
                            ce_matrix, _ = self.ce_agent.estimate_plan_batch(
                                instruction=instruction,
                                plan=rewritten_plan,
                                candidate_backbones=LLMBackbone.available_backbones(),
                            )
                            usage_after_ce = self.ce_agent.caller.get_total_usage()
                            ce_metrics = {
                                "prompt_tokens": usage_after_ce.get("input_tokens", 0),
                                "completion_tokens": usage_after_ce.get("output_tokens", 0),
                                "cache_read_tokens": usage_after_ce.get("cached_tokens", 0),
                                "reasoning_tokens": usage_after_ce.get("reasoning_tokens", 0),
                                "total_tokens": usage_after_ce.get("total_tokens", 0),
                                "accumulated_cost": 0.0,
                                "llm_backbone": self.ce_agent._get_llm_backbone(),
                            }
                        except Exception as e:
                            logger.warning(f"  Re-CE failed: {e}, stopping iteration")
                            break

                    logger.info(f"  Plan after round {round_idx + 1}: {len(rewritten_plan.operators)} operators")
                    for op in rewritten_plan.operators:
                        bb_str = op.llm_backbone.value if op.llm_backbone else "unassigned"
                        logger.info(f"    Op {op.index}: [{bb_str}] {op.subtask[:80]}")

            else:
                for op in rewritten_plan.operators:
                    cheap_ce = ce_matrix.get(op.index, {}).get("CHEAP")
                    if cheap_ce and not cheap_ce.error and cheap_ce.uncertainty >= 0.35:
                        exp_ce = ce_matrix.get(op.index, {}).get("EXPENSIVE")
                        if exp_ce and not exp_ce.error and exp_ce.uncertainty < cheap_ce.uncertainty:
                            op.llm_backbone = LLMBackbone.EXPENSIVE
                            logger.info(
                                f"  Op {op.index}: upgrading CHEAP→EXPENSIVE "
                                f"(cheap_u={cheap_ce.uncertainty:.2f} → exp_u={exp_ce.uncertainty:.2f})"
                            )

        elif not self.use_ce or not self.ce_agent:
            for op in rewritten_plan.operators:
                if op.llm_backbone is None:
                    op.llm_backbone = LLMBackbone.CHEAP

        self._write_json(os.path.join(logs_dir, "final_plan.json"), rewritten_plan.to_dict_list())
        logger.info(f"[OperatorPipeline] Final plan: {len(rewritten_plan.operators)} operators")
        for op in rewritten_plan.operators:
            bb_str = op.llm_backbone.value if op.llm_backbone else "unassigned"
            logger.info(f"  Op {op.index}: [{bb_str}] {op.subtask[:80]}")

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

        def _add_metrics_to_backbone_summary(
            summary: dict[str, dict[str, Any]],
            metrics: dict[str, Any],
        ):
            if not metrics or not metrics.get("llm_backbone"):
                return
            bb = metrics["llm_backbone"]
            if bb not in summary:
                summary[bb] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "cache_read_tokens": 0, "reasoning_tokens": 0,
                    "total_tokens": 0, "accumulated_cost": 0.0,
                    "operator_count": 0,
                }
            for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                       "reasoning_tokens", "total_tokens"):
                summary[bb][k] += metrics.get(k, 0)
            summary[bb]["accumulated_cost"] += metrics.get("accumulated_cost", 0.0)
            summary[bb]["operator_count"] += 1

        by_backbone: dict[str, dict[str, Any]] = {}
        for bb_key, bb_val in exec_total_metrics["by_backbone"].items():
            by_backbone[bb_key] = dict(bb_val)
        _add_metrics_to_backbone_summary(by_backbone, planning_metrics)
        _add_metrics_to_backbone_summary(by_backbone, ce_metrics)
        _add_metrics_to_backbone_summary(by_backbone, rewrite_metrics)

        all_phase_metrics = [planning_metrics, ce_metrics, rewrite_metrics, exec_total_metrics["total"]]
        total_tokens = {
            "prompt_tokens": sum(m.get("prompt_tokens", 0) for m in all_phase_metrics),
            "completion_tokens": sum(m.get("completion_tokens", 0) for m in all_phase_metrics),
            "cache_read_tokens": sum(m.get("cache_read_tokens", 0) for m in all_phase_metrics),
            "reasoning_tokens": sum(m.get("reasoning_tokens", 0) for m in all_phase_metrics),
            "total_tokens": sum(m.get("total_tokens", 0) for m in all_phase_metrics),
            "accumulated_cost": sum(m.get("accumulated_cost", 0.0) for m in all_phase_metrics),
        }

        total_metrics = {
            "planning": planning_metrics,
            "cost_estimation": ce_metrics,
            "rewrite": rewrite_metrics,
            "execution": exec_total_metrics["total"],
            "execution_by_backbone": exec_total_metrics["by_backbone"],
            "by_backbone": by_backbone,
            "total": total_tokens,
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
            bb_str = op.llm_backbone.value if op.llm_backbone else "unassigned"
            log_lines.append(f"- Op {op.index}: [{bb_str}] {op.subtask}")
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
    def _validate_information_closure(plan: OperatorPlan) -> OperatorPlan:
        available_info: set[str] = set()
        fixed_plan = copy.deepcopy(plan)
        insertions: list[tuple[int, Operator]] = []

        for i, op in enumerate(fixed_plan.operators):
            if not op.requires_info:
                available_info.update(op.produces_info)
                continue

            missing = [info for info in op.requires_info if info not in available_info]

            if missing and op.cognitive_type != CognitiveType.PERCEPTION:
                patch_op = Operator(
                    index=0,
                    subtask=f"Gather required information: {'; '.join(missing)}",
                    cognitive_type=CognitiveType.PERCEPTION,
                    requires_info=[],
                    produces_info=missing,
                )
                insertions.append((i, patch_op))
                logger.info(
                    f"  [info-closure] Op {op.index} missing: {missing}, inserting perception op"
                )
                available_info.update(missing)

            available_info.update(op.produces_info)

        for pos, new_op in reversed(insertions):
            fixed_plan.operators.insert(pos, new_op)

        if insertions:
            fixed_plan.reindex()
            logger.info(
                f"[OperatorPipeline] Info closure: inserted {len(insertions)} perception operators, "
                f"total now {len(fixed_plan.operators)}"
            )

        return fixed_plan

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
