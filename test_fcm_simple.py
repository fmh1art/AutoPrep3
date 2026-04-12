"""
测试FCM Agent的简单脚本
"""

import yaml
from src.benchmarks.swe_bench_runner import SweBenchRunner
from openhands.sdk import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # 加载配置
    cfg = yaml.safe_load(open("_config/doubao.yaml"))
    cheap_cfg = yaml.safe_load(open("_config/doubao_flash.yaml"))

    # 创建runner with FCM
    runner = SweBenchRunner(
        exp_cfg=cfg,
        cheap_exp_cfg=cheap_cfg,
        tmp_root="./_tmp/fcm_test",
        prompt_path="./src/prompts/react_baseline.j2",
        use_fcm=True,
    )

    # 测试一个实例
    instances = runner.prepare_instances(
        dataset="../_AutpPrep3_out/_data/SWEBenchVerified",
        split="test",
        eval_limit=1,
    )

    logger.info(f"Testing FCM with {len(instances)} instance")

    for instance in instances:
        workspace = None
        try:
            workspace = runner.prepare_workspace(instance)
            result = runner.evaluate_instance(instance, workspace)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if workspace:
                workspace.cleanup()
