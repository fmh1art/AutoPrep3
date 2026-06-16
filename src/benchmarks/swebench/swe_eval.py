"""
SWE-bench 原生 evaluation 逻辑。

直接使用 swebench.harness 的 API，在官方 SWE-bench Docker 镜像中：
  1. 构建 TestSpec
  2. 创建并启动容器
  3. 应用 git patch
  4. 执行 eval.sh 跑测试
  5. 用 get_eval_report 解析结果
  6. 清理容器
"""

from __future__ import annotations

import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import docker

from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_build import (
    build_container,
    close_logger,
    setup_logger,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

# 与 run_evaluation.py 一致的三种 patch 应用命令（优先级从高到低）
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

# 官方 SWE-bench 镜像前缀（namespace=swebench 时的远程拉取格式）
SWEBENCH_IMAGE_PREFIX = "swebench"


def get_swebench_image(instance_id: str, tag: str = "latest") -> str:
    """
    根据 instance_id 返回官方 SWE-bench Docker 镜像名。

    例如：
        astropy__astropy-11693 → swebench/sweb.eval.x86_64.astropy_1776_astropy-11693:latest
        django__django-12345   → swebench/sweb.eval.x86_64.django_1776_django-12345:latest
    """
    repo, name = instance_id.split("__")
    return (
        f"{SWEBENCH_IMAGE_PREFIX}/sweb.eval.x86_64."
        f"{repo}_1776_{name}:{tag}"
    )


def run_swebench_eval(
    instance: dict[str, Any],
    git_patch: str,
    run_id: str | None = None,
    tmp_dir: str = "/tmp",
    timeout: int = 1800,
    model_name_or_path: str = "agent",
    skip_completed: bool = True,
) -> dict[str, Any]:
    """
    在 SWE-bench 官方镜像中运行评估，直接使用 swebench.harness API。
    与官方 run_evaluation.py 流程完全对齐。

    Args:
        instance:  SWE-bench 数据集中的单条记录（dict），需包含以下字段：
                   instance_id, repo, version, base_commit, test_patch,
                   FAIL_TO_PASS, PASS_TO_PASS
        git_patch: LLM agent 生成的 git diff patch 字符串（可为空）
        run_id:    运行 ID，用于容器命名和日志目录；默认随机生成
        tmp_dir:   本地临时目录，用于存放日志和 patch 文件
        timeout:   eval.sh 执行超时秒数
        model_name_or_path: 模型名称，用于日志目录结构
        skip_completed: 如果 report.json 已存在则跳过该实例

    Returns:
        dict，包含：
          - resolved (bool)       : 是否全部 FAIL_TO_PASS 测试通过
          - patch_applied (bool)  : patch 是否成功应用
          - timed_out (bool)      : 是否超时（可选）
          - report (dict)         : get_eval_report 返回的完整报告
          - error (str | None)    : 异常信息（如有）
          - skipped (bool)        : 是否因已完成而跳过
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    instance_id = instance["instance_id"]
    client = docker.from_env()

    # ── 1. 空 patch 检查（与官方对齐：空 patch 不运行评估）───────────────────
    if git_patch is None or git_patch.strip() == "":
        return {
            "instance_id": instance_id,
            "resolved": False,
            "patch_applied": False,
            "timed_out": False,
            "report": {},
            "error": "empty patch",
            "skipped": True,
        }

    # TestSpec：namespace="swebench" 表示使用远程镜像（直接 pull）
    # 与官方对齐：显式指定 arch="x86_64"
    test_spec = make_test_spec(
        instance,
        namespace=SWEBENCH_IMAGE_PREFIX,
        arch="x86_64",
    )

    # 日志目录结构与官方对齐：logs/run_evaluation/{run_id}/{model_name}/{instance_id}/
    model_name_safe = model_name_or_path.replace("/", "__")
    log_dir = Path(tmp_dir) / RUN_EVALUATION_LOG_DIR / run_id / model_name_safe / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_instance.log"
    report_path = log_dir / LOG_REPORT
    logger = setup_logger(instance_id, log_file)

    # ── 2. 已完成实例跳过（与官方对齐）──────────────────────────────────────
    if skip_completed and report_path.exists():
        import json
        report = json.loads(report_path.read_text())
        resolved = report.get(instance_id, {}).get("resolved", False)
        logger.info(f"Skipping {instance_id} - already completed, resolved={resolved}")
        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "patch_applied": True,
            "timed_out": False,
            "report": report,
            "error": None,
            "skipped": True,
        }

    container = None
    try:
        # ── 1. 构建容器（如镜像不存在则自动 pull）─────────────────────────
        container = build_container(
            test_spec,
            client,
            run_id,
            logger,
            nocache=False,
            force_rebuild=False,
        )
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # ── 2. 写入并应用 patch ────────────────────────────────────────────
        patch_file = log_dir / "patch.diff"
        patch_file.write_text(git_patch or "", encoding=UTF8)
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        applied_patch = False
        for cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"Applied Patch:\n{val.output.decode(UTF8)}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {cmd}")

        if not applied_patch:
            logger.info(f">>>>> Patch Apply Failed:\n{val.output.decode(UTF8)}")
            return {
                "instance_id": instance_id,
                "resolved": False,
                "patch_applied": False,
                "timed_out": False,
                "report": {},
                "error": "patch apply failed",
                "skipped": False,
            }

        # ── 3. 获取 eval.sh 执行前的 git diff（与官方对齐）───────────────────
        git_diff_output_before = (
            container.exec_run(
                "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
            )
            .output.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        # ── 4. 写入并执行 eval.sh ──────────────────────────────────────────
        eval_file = log_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script, encoding=UTF8)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        test_output, timed_out, runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {runtime:.2f} seconds")
        with open(test_output_path, "w", encoding=UTF8) as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")

        if timed_out:
            return {
                "instance_id": instance_id,
                "resolved": False,
                "patch_applied": True,
                "timed_out": True,
                "report": {},
                "error": f"Test timed out after {timeout} seconds.",
                "skipped": False,
            }

        # ── 5. 获取 eval.sh 执行后的 git diff 并对比（与官方对齐）────────────
        git_diff_output_after = (
            container.exec_run(
                "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
            )
            .output.decode(UTF8)
            .strip()
        )
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # ── 6. 解析结果 ────────────────────────────────────────────────────
        logger.info(f"Grading answer for {instance_id}...")
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: git_patch,
            KEY_MODEL: model_name_or_path,
        }
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {resolved}"
        )

        # ── 7. 写入 report.json（与官方对齐）────────────────────────────────
        with open(report_path, "w", encoding=UTF8) as f:
            import json
            f.write(json.dumps(report, indent=4))

        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "patch_applied": True,
            "timed_out": False,
            "report": report,
            "error": None,
            "skipped": False,
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(f"Error in evaluating model for {instance_id}: {e}\n{err}")
        return {
            "instance_id": instance_id,
            "resolved": False,
            "patch_applied": False,
            "timed_out": False,
            "report": {},
            "error": str(e),
            "skipped": False,
        }

    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)
