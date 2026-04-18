#!/usr/bin/env python
"""
直接测试 CustomCodeAgent 类的简单脚本

这个脚本不依赖完整的 SWE-bench 流程，直接创建一个简单的任务测试自定义 Agent。
"""

import os
import sys
import yaml
import logging

from openhands.workspace import DockerWorkspace

from src.agent.custom_code_agent import CustomCodeAgent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # 1. 加载配置
    config_path = "_config/doubao.yaml"
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return 1
    
    cfg = yaml.safe_load(open(config_path))
    
    # 2. 创建自定义 CodeAgent
    logger.info("创建 CustomCodeAgent...")
    agent = CustomCodeAgent(
        llm_cfg=cfg,
        max_steps=20,  # 限制步数，快速测试
    )
    
    # 3. 准备 Docker 工作空间
    logger.info("启动 DockerWorkspace...")
    workspace = None
    try:
        workspace = DockerWorkspace(
            server_image="ghcr.io/openhands/agent-server:latest-python",
            platform="linux/amd64",  # 根据你的架构修改
        )
        logger.info("DockerWorkspace 启动成功")
        
        # 4. 定义一个简单的测试任务
        task = """\
请帮我完成以下任务：
1. 查看 /workspace 目录
2. 创建一个名为 test.txt 的文件，内容为 "Hello, CustomCodeAgent!"
3. 查看这个文件内容
4. 完成任务

注意：所有操作都在 /workspace 目录下进行。
"""
        
        # 5. 运行 Agent
        logger.info("开始运行 Agent...")
        output_dir = "./_tmp/test_custom_agent_simple"
        os.makedirs(output_dir, exist_ok=True)
        
        result = agent.run(
            instruction=task,
            workspace=workspace,
            output_dir=output_dir,
        )
        
        # 6. 打印结果
        logger.info("\n" + "="*60)
        logger.info("Agent 运行完成！")
        logger.info(f"完成消息: {result.finish_message}")
        logger.info(f"总步数: {len(result.trajectory)}")
        logger.info(f"Metrics: {result.metrics}")
        logger.info(f"\n详细输出已保存到: {output_dir}")
        
        return 0
        
    except Exception as e:
        import traceback
        logger.error(f"出错: {e}")
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        if workspace is not None:
            try:
                logger.info("清理 DockerWorkspace...")
                workspace.cleanup()
            except Exception as e:
                logger.warning(f"清理失败: {e}")


if __name__ == "__main__":
    sys.exit(main())
