"""
测试ForwardContextManagement Agent的简单脚本
"""

import yaml
import time
from src.benchmarks.swe_bench_runner import SweBenchRunner
from openhands.sdk import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # 加载配置
    cfg = yaml.safe_load(open("_config/doubao.yaml"))
    cheap_cfg = yaml.safe_load(open("_config/doubao_flash.yaml"))

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tmp_root = f"./_tmp/fcm_test_{timestamp}"

    # 代理配置
    HTTP_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
    NO_PROXY_LIST = "localhost,127.0.0.1,::1,bytedance.net,byted.org"

    # 创建runner，启用FCM
    runner = SweBenchRunner(
        exp_cfg=cfg,
        cheap_exp_cfg=cheap_cfg,
        tmp_root=tmp_root,
        prompt_path="./src/prompts/code_agent_fcm.j2",
        http_proxy=HTTP_PROXY,
        no_proxy=NO_PROXY_LIST,
        use_fcm=True,  # 启用FCM
    )

    # 加载一个测试实例
    all_instances = runner.prepare_instances(
        dataset="../_AutpPrep3_out/_data/SWEBenchVerified",
        split="test",
        eval_limit=1,  # 只测试一个
    )

    logger.info(f"Testing FCM agent with {len(all_instances)} instance(s)")

    for instance in all_instances:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {instance['instance_id']}")
        workspace = None
        try:
            workspace = runner.prepare_workspace(instance)
            result = runner.evaluate_instance(instance, workspace)
            logger.info(f"Result: resolved={result.get('resolved', False)}")
            logger.info(f"Metrics: {result.get('metrics', {})}")
        except Exception as e:
            import traceback
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
        finally:
            if workspace is not None:
                try:
                    workspace.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Workspace cleanup failed: {cleanup_error}")
