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
) -> dict[str, Any]:
    """
    在 SWE-bench 官方镜像中运行评估，直接使用 swebench.harness API。

    Args:
        instance:  SWE-bench 数据集中的单条记录（dict），需包含以下字段：
                   instance_id, repo, version, base_commit, test_patch,
                   FAIL_TO_PASS, PASS_TO_PASS
        git_patch: LLM agent 生成的 git diff patch 字符串（可为空）
        run_id:    运行 ID，用于容器命名和日志目录；默认随机生成
        tmp_dir:   本地临时目录，用于存放日志和 patch 文件
        timeout:   eval.sh 执行超时秒数

    Returns:
        dict，包含：
          - resolved (bool)       : 是否全部 FAIL_TO_PASS 测试通过
          - patch_applied (bool)  : patch 是否成功应用
          - timed_out (bool)      : 是否超时（可选）
          - report (dict)         : get_eval_report 返回的完整报告
          - error (str | None)    : 异常信息（如有）
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    instance_id = instance["instance_id"]
    client = docker.from_env()

    # TestSpec：namespace="swebench" 表示使用远程镜像（直接 pull）
    test_spec = make_test_spec(instance, namespace=SWEBENCH_IMAGE_PREFIX)

    # 日志目录
    log_dir = Path(tmp_dir) / "swe_eval_logs" / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval.log"
    logger = setup_logger(instance_id, log_file)

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
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        applied_patch = False
        for cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"Patch applied with: {cmd}")
                applied_patch = True
                break
            else:
                logger.info(f"Patch apply failed with '{cmd}': {val.output.decode(UTF8)}")

        if not applied_patch:
            logger.info(f"All patch apply commands failed for {instance_id}")
            return {
                "resolved": False,
                "patch_applied": False,
                "report": {},
                "error": "patch apply failed",
            }

        # ── 3. 写入并执行 eval.sh ──────────────────────────────────────────
        eval_file = log_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script, encoding=UTF8)
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        test_output, timed_out, runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        logger.info(f"Test runtime: {runtime:.2f}s, timed_out={timed_out}")

        test_output_path = log_dir / "test_output.txt"
        test_output_path.write_text(test_output, encoding=UTF8)

        if timed_out:
            return {
                "resolved": False,
                "patch_applied": True,
                "timed_out": True,
                "report": {},
                "error": f"eval timed out after {timeout}s",
            }

        # ── 4. 解析结果 ────────────────────────────────────────────────────
        prediction = {
            "instance_id": instance_id,
            "model_patch": git_patch,
            "model_name_or_path": "agent",
        }
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)
        logger.info(f"Eval result for {instance_id}: resolved={resolved}")

        return {
            "resolved": resolved,
            "patch_applied": True,
            "timed_out": False,
            "report": report,
            "error": None,
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(f"Error in run_swebench_eval for {instance_id}: {e}\n{err}")
        return {
            "resolved": False,
            "patch_applied": False,
            "timed_out": False,
            "report": {},
            "error": str(e),
        }

    finally:
        cleanup_container(client, container, logger)
        close_logger(logger)
