# 主要需求

请实现/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py类，要求实现planning agent。

要求如下：

- 使用给定的prompt：
	- /home/fanmeihao/projects/AutoPrep3/src/prompts/code_agent_plan_mode_planning.j2
	- /home/fanmeihao/projects/AutoPrep3/src/prompts/code_agent_plan_mode_execution.j2
- 要求统计两个agent消耗的token总数（包括input token，output token，cache token等）
- 针对execution agent，要求记录每一步消耗的token数。
- 最后输出的文件目录为：
	- {output_dir}/log下记录log文件，包括：
		- log.md为logger的输出文件，planner_trajectory.md和execution_trajectory.md为两个agent工作过程中生成的trajectory。
		- 记录token消耗的json文件
	- {output_dir}/swe_eval_logs文件，为evaluation的输出结果

可参考的SDK example文件：
/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/examples/01_standalone_sdk/24_planning_agent_workflow.py
/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/examples/01_standalone_sdk/05_use_llm_registry.py
此外，遇到不理解的地方，你可以阅读/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/这个文件夹下的任意文件

对于任何不确定的接口，你可以直接阅读sdk源码。但是确保每个接口都是正确的，并且你能理解其实现原理的。

# Referred Source SDK files
(all possible sdk files you need to read!)

- 示例脚本：
  - `/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/examples/01_standalone_sdk/24_planning_agent_workflow.py`
  - `/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/examples/01_standalone_sdk/05_use_llm_registry.py`
- SDK 源码根目录（可阅读任意文件以确认接口与实现细节）：`/home/fanmeihao/projects/_AutpPrep3_out/openhands-software-agent-sdk/`
- 本项目模板文件：
  - `/home/fanmeihao/projects/AutoPrep3/src/prompts/code_agent_plan_mode_planning.j2`
  - `/home/fanmeihao/projects/AutoPrep3/src/prompts/code_agent_plan_mode_execution.j2`
- 本项目核心实现文件与函数参考：
  - `CodeAgentPlanMode.run`入口：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:290`
  - `_extract_metrics`：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:85`
  - `_extract_token_usage_items`：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:107`
  - `_build_execution_trajectory`：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:205`
  - `_extract_execution_steps_metrics`：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:218`
  - `_add_metrics`与`_merge_metrics`：`/home/fanmeihao/projects/AutoPrep3/src/agent/code_agent_plan_mode.py:135`、`:65`


# 当前版本的问题

_tmp/parallel_2026-04-03_17-00-01 目录记录了模型的输出，现在的问题是token记录的文件：

_tmp/parallel_2026-04-03_17-00-01/log/token_usage.json
_tmp/parallel_2026-04-03_17-00-01/results.json

里面没有统计step层面的token消耗，并且当前token数都是0。请阅读SDK源码文件，分析原因，并找到正确的接口。