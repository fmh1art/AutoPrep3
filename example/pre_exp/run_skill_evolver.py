import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from openhands.workspace import DockerWorkspace
from src.agent.skill_evolver import SkillEvolver

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str, fallback_base: Path | None = None) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    if fallback_base is not None:
        return str((fallback_base / p).resolve())
    return str((REPO_ROOT / p).resolve())


def load_trajectories(input_dir: str, max_cnt = 9) -> list[dict]:
    log_dir = os.path.join(input_dir, "log")
    if not os.path.isdir(log_dir):
        logger.error(f"log directory not found: {log_dir}")
        return []

    trajectories = []
    for case_id in sorted(os.listdir(log_dir)):
        if len(trajectories) > max_cnt and max_cnt > 0:
            break
        
        case_dir = os.path.join(log_dir, case_id)
        if not os.path.isdir(case_dir):
            continue

        messages_path = os.path.join(case_dir, "messages.jsonl")
        records_path = os.path.join(case_dir, "records.json")

        if not os.path.isfile(messages_path) or not os.path.isfile(records_path):
            logger.warning(f"skipping {case_id}: missing messages.jsonl or records.json")
            continue

        try:
            messages = []
            with open(messages_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))

            with open(records_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            step2message = records.get("step2message", {})
            step2message['0'] = [1]

            step2meta_path = os.path.join(case_dir, "step2meta_info.json")
            step2meta = {}
            if os.path.isfile(step2meta_path):
                with open(step2meta_path, "r", encoding="utf-8") as f:
                    step2meta = json.load(f)

            trajectories.append({
                "messages": messages,
                "step2message": step2message,
                "step2meta": step2meta,
            })
            meta_flag = "with DAG" if step2meta else "no DAG"
            logger.info(f"loaded trajectory: {case_id} ({len(messages)} messages, {meta_flag})")
        except Exception as e:
            logger.warning(f"failed to load {case_id}: {e}")

    return trajectories


def create_workspace(agent_server_image: str = "ghcr.io/openhands/agent-server:latest-python") -> DockerWorkspace:
    workspace = DockerWorkspace(
        server_image=agent_server_image,
        working_dir="/workspace",
    )
    workspace.execute_command("mkdir -p /workspace/.skill/meta_skill /workspace/.skill/tool_skill", timeout=30)
    meta_skill_path = "/workspace/.skill/meta_skill/meta_skill.md"
    workspace.execute_command(
        f"test -f {meta_skill_path} || echo '# Meta Skills\n' > {meta_skill_path}",
        timeout=30,
    )
    return workspace


def main():
    parser = argparse.ArgumentParser(description="Run SkillEvolver on existing experiment logs")
    parser.add_argument("--input-dir", type=str, default="_tmp/code_agent_limit64_doubao_2026-05-16_18-59-41", help="实验输出目录")
    parser.add_argument("--exp-config", type=str, required=True, help="模型配置文件(yaml)")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认为 ./_tmp/skill_evolver_{exp_name}_{timestamp}")
    parser.add_argument("--trajectory-cnt", type=int, default=3, help="每次evolve使用的trajectory数量")
    parser.add_argument("--max-cnt", type=int, default=-1, help="加载trajectory的最大数量")
    parser.add_argument("--evolve-epoch", type=int, default=1, help="evolve轮数")
    parser.add_argument("--max-step", type=int, default=200, help="Agent最大步数")
    parser.add_argument("--agent-server-image", type=str, default="ghcr.io/openhands/agent-server:latest-python", help="Docker workspace镜像")
    parser.add_argument("--debug", action="store_true", help="启用DEBUG级别日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.exp_config = _resolve_path(args.exp_config)
    args.input_dir = _resolve_path(args.input_dir)

    with open(args.exp_config, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/skill_evolver_{exp_name}_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading trajectories from {args.input_dir}")
    trajectories = load_trajectories(args.input_dir, max_cnt=args.max_cnt)
    if not trajectories:
        logger.error("No trajectories found, exiting")
        sys.exit(1)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    workspace = None
    try:
        logger.info("Creating Docker workspace...")
        workspace = create_workspace(agent_server_image=args.agent_server_image)

        evolver = SkillEvolver(
            llm_cfg=llm_cfg,
            output_dir=args.output_dir,
            trajectory_cnt=args.trajectory_cnt,
            evolve_epoch=args.evolve_epoch,
            max_step=args.max_step,
        )

        logger.info("Starting skill evolution...")
        evolver.evolve(tol_trajectories=trajectories, workspace=workspace)
        logger.info("Skill evolution completed")
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as e:
                logger.warning(f"Workspace cleanup failed: {e}")


if __name__ == "__main__":
    main()
