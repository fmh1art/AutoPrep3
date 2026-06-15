"""并行运行 SWE-bench 的 PyCodeGraph 版本。"""

try:
    from benchmark_code_agent import main
except ModuleNotFoundError:
    from example.benchmark_code_agent import main


if __name__ == "__main__":
    main(enable_pycodegraph=True)

# 运行示例：
# python /home/fanmeihao/projects/AutoPrep3_0526/example/benchmark_code_agent_with_pycodegraph.py \
#   --dataset /path/to/dataset \
#   --split test \
#   --eval-limit 10 \
#   --exp-config /home/fanmeihao/projects/AutoPrep3_0526/_config/deepseekv4_flash.yaml \
#   --max-steps 200 \
#   --parallel 4 \
#   --output-dir /home/fanmeihao/projects/AutoPrep3_0526/_tmp/pycodegraph_demo_run



"""
# 运行示例：
python example/benchmark_code_agent_with_pycodegraph.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --max-steps 200 \
  --parallel 8 \


python example/benchmark_code_agent_with_pycodegraph.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/deepseekv4_flash.yaml \
  --max-steps 200 \
  --parallel 8 \
"""