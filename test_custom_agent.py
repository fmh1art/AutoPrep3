#!/usr/bin/env python
"""
测试自定义 CodeAgent 的运行脚本

使用方法:
    python test_custom_agent.py
"""

import os
import sys
import time
import json
import uuid
import logging
import yaml

from src.benchmarks.swe_bench_runner import SweBenchRunner

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # 1. 加载配置文件
    config_path = "_config/doubao.yaml"
    cheap_config_path = "_config/doubao_flash.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return 1
    
    cfg = yaml.safe_load(open(config_path))
    cheap_cfg = yaml.safe_load(open(cheap_config_path))
    
    # 2. 准备输出目录
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    tmp_root = f"./_tmp/test_custom_agent_{timestamp}"
    os.makedirs(tmp_root, exist_ok=True)
    logger.info(f"输出目录: {tmp_root}")
    
    # 3. 代理配置（根据你的环境修改）
    HTTP_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
    NO_PROXY_LIST = "localhost,127.0.0.1,::1,bytedance.net,byted.org"
    
    # 4. 创建 SweBenchRunner，启用自定义 Agent
    logger.info("创建 SweBenchRunner（启用 CustomCodeAgent）")
    runner = SweBenchRunner(
        exp_cfg=cfg,
        cheap_exp_cfg=cheap_cfg,
        tmp_root=tmp_root,
        prompt_path="./src/prompts/query.j2",
        http_proxy=HTTP_PROXY,
        no_proxy=NO_PROXY_LIST,
        use_custom_agent=True,  # 关键：启用自定义 CodeAgent
    )
    
    # 5. 加载测试实例
    # 可选：加载 SWE-bench 数据集
    logger.info("加载测试实例")
    try:
        all_instances = runner.prepare_instances(
            dataset="../_AutpPrep3_out/_data/SWEBenchVerified",
            split="test",
            eval_limit=1,  # 只测试 1 个实例
        )
    except Exception as e:
        logger.warning(f"无法加载 SWE-bench 数据集: {e}")
        logger.info("创建一个简单的测试实例...")
        
        # 如果没有 SWE-bench 数据，创建一个简单的测试实例
        all_instances = [
            {
                "instance_id": "test_001",
                "repo": "python/cpython",
                "base_commit": "v3.9.0",
                "problem_statement": "这是一个简单的测试任务，验证自定义 CodeAgent 能否正常工作。",
            }
        ]
    
    logger.info(f"加载到 {len(all_instances)} 个实例")
    
    # 6. 运行测试
    for idx, instance in enumerate(all_instances):
        logger.info(f"\n{'='*60}")
        logger.info(f"测试实例 [{idx+1}/{len(all_instances)}]: {instance['instance_id']}")
        
        workspace = None
        try:
            # 准备工作空间
            logger.info("启动 Docker 工作空间...")
            workspace = runner.prepare_workspace(instance)
            logger.info("Docker 工作空间启动成功")
            
            # 运行评估
            logger.info("开始运行自定义 CodeAgent...")
            result = runner.evaluate_instance(instance, workspace)
            
            # 打印结果
            logger.info("测试完成！")
            logger.info(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            import traceback
            logger.error(f"测试出错: {e}")
            logger.error(traceback.format_exc())
            
            # 保存错误信息
            instance_id = instance.get("instance_id", "unknown")
            run_id = str(uuid.uuid4())[:8]
            log_dir = os.path.join(tmp_root, "log", run_id)
            os.makedirs(log_dir, exist_ok=True)
            
            error_result = {
                "instance_id": instance_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            
            error_file = os.path.join(log_dir, f"{instance_id}_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            logger.info(f"错误信息已保存到: {error_file}")
            
        finally:
            if workspace is not None:
                try:
                    logger.info("清理工作空间...")
                    workspace.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"工作空间清理失败: {cleanup_error}")
    
    logger.info(f"\n{'='*60}")
    logger.info("全部测试完成！")
    logger.info(f"输出目录: {tmp_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
