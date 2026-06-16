import copy
import json
import logging
import os
import time

from openhands.workspace import DockerWorkspace

from src.element.executor import CustomizedCodeAgentExecutor
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import compute_metrics, save_jsonl, setup_file_logger, write_messages_to_markdown_format

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_STEP = 5

SYSTEM_PROMPT = """You are an agent, a helpful AI assistant that use pre-defined tools to solve tasks.""".strip()

PYCODEGRAPH_PROMPT = """
This workspace has been pre-indexed with **PyCodeGraph**, a lightweight, zero-dependency
local code-intelligence knowledge graph (Python symbols, references, containment)
stored at `<repo_path>/.pycodegraph/graph.json`.

Prefer PyCodeGraph as a candidate-narrowing tool, not as the final source of truth.
Use it to quickly find likely symbols, then read the real file content before editing.

You have the following dedicated tools (all require `repo_path`):

- `pycodegraph_search(query)` — find symbol definitions by name/node_id. Returns
  ranked candidate definitions with file, line range, kind, stable id, and match reasons.
- `pycodegraph_read_symbol(symbol)` — read the most relevant definition with surrounding
  code snippet. Use this when search returns multiple candidates or when you need real context.
- `pycodegraph_callers(symbol)` — list symbols that REFERENCE the given symbol
  (what calls/uses it). Useful before editing.

These tools are intentionally minimal. Index maintenance is handled internally; no separate
status/init step is required during normal task solving.

**CRITICAL RULES for PyCodeGraph:**

1. Start with `pycodegraph_search` to narrow candidate definitions.
2. If the results are ambiguous, use `pycodegraph_read_symbol` before editing.
3. Use `pycodegraph_callers` only when direct usages matter for the change.
4. Do not assume graph results are perfect; confirm by reading actual file content when needed.
5. repo_path is the absolute path of the repo root inside the container, e.g.
   `/workspace/<repo_name>` (provided in the task prompt).
""".strip()


TASK_PROMPT = """
You need to solve the following task:

{task}
""".strip()

DEFAULT_CONTINUE_PROMPT = (
    "Please continue working on the task on whatever approach you think is suitable.\n"
    "When you think you have solved the question, please use the finish tool and "
    "include your final answer in the message parameter of the finish tool.\n"
    "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
)


def handle_no_tool_calls(
    fake_response_count: int,
    max_fake_responses: int,
    continue_prompt: str = DEFAULT_CONTINUE_PROMPT,
) -> tuple[list[dict], bool]:
    if fake_response_count >= max_fake_responses:
        return [{"role": "user", "content": "Agent terminated: exceeded maximum consecutive responses without tool calls. Use the finish tool to end the interaction."}], True
    msg = continue_prompt
    if fake_response_count >= 2:
        msg += 'If you want to give up, use the "finish" tool to finish the interaction.\n'
    return [{"role": "user", "content": msg}], False


class CustomizedCodeAgent:
    
    def __init__(self, llm_cfg: dict, output_dir: str, max_step:int=200, use_pycodegraph: bool = False):
        self.llm = SimpleAPICaller(
            llm_name=llm_cfg.get('llm_name', None),
            api_key=llm_cfg.get('key', None),
            base_url=llm_cfg.get('openai_base_url', None),
        )
        self.MAX_STEP = max_step
        self.MAX_RETRY_LLM_CALL = 3
        self.output_dir = output_dir
        self.use_pycodegraph = use_pycodegraph
        self.executor = CustomizedCodeAgentExecutor(use_pycodegraph=use_pycodegraph)
        self.tools = self.executor.get_tool_definitions()
        self._fake_response_count = 0
        self._max_fake_responses = 10
        self._setup_logger()

    def _setup_logger(self):
        setup_file_logger(logger, self.output_dir, "code_agent.log")

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, self.MAX_RETRY_LLM_CALL + 1):
            try:
                start_usage = self.llm.get_total_usage()
                start_time = time.time()
                response_dict = self.llm.chat(
                    messages=messages,
                    tools=self.tools,
                )
                end_usage = self.llm.get_total_usage()
                end_time = time.time()
                use_token = compute_metrics(start_usage, end_usage)
                use_time = end_time - start_time
                messages.append(response_dict)
                return response_dict, use_token, use_time
            except Exception as e:
                last_exc = e
                logger.error(f"[CodeAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"LLM call failed after {self.MAX_RETRY_LLM_CALL} attempts"
        ) from last_exc
        
    def act(self, messages: list[dict]):
        response_dict, use_token, use_time = self._call_llm(messages)
        return response_dict, messages, use_token, use_time
    
    def observe(self, response_dict, workspace: DockerWorkspace, step: int = 0):
        tool_calls = response_dict.get("tool_calls") or []
        if not tool_calls:
            self._fake_response_count += 1
            return handle_no_tool_calls(self._fake_response_count, self._max_fake_responses)

        observations = []
        finished = False
        executed_count = 0
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                observations.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: Failed to parse tool arguments for '{tool_name}': {tc['function']['arguments']}\n[Step {step}]",
                })
                continue

            if tool_name == "finish":
                observation = tool_args.get("message", "") or "Agent finished."
                finished = True
            else:
                executed_count += 1
                if executed_count > MAX_TOOL_CALLS_PER_STEP:
                    observation = f"Skipped: max {MAX_TOOL_CALLS_PER_STEP} tool calls per step exceeded."
                else:
                    observation = self.executor.execute_tool(tool_name, tool_args, workspace)

            observations.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"{observation}\n[Step {step}]",
            })
            if finished:
                break
        return observations, finished
    
    def save_current_states(self, step, records: dict, messages: list[dict], use_token, use_time, response_dict, observations, step2message: dict | None = None):
        metrics = copy.deepcopy(use_token) if use_token else {}
        metrics['use_time'] = use_time or 0.0

        record_md = write_messages_to_markdown_format(messages) 

        current_tool = response_dict.get("tool_calls") or []

        step_record = {
            'step': step,
            'metrics': metrics,
            'current_tool': current_tool,
            'observation': observations,
        }
        if step2message is not None:
            step_record['message_indices'] = step2message.get(step, [])
        records['steps'].append(step_record)

        os.makedirs(f"{self.output_dir}/log", exist_ok=True)
        with open(f"{self.output_dir}/log/llm_step.md", "w") as f:
            f.write(record_md)
        with open(f"{self.output_dir}/records.json", "w") as f:
            json.dump(records, f, indent=4, default=str)
        
    def run(self, task_instruction: str, workspace: DockerWorkspace, messages: list[dict]=None, begin_step: int = 0):
        self._fake_response_count = 0
        system_prompt = SYSTEM_PROMPT
        if self.use_pycodegraph:
            system_prompt = system_prompt + "\n\n" + PYCODEGRAPH_PROMPT

        if messages is None:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': TASK_PROMPT.format(task=task_instruction)},
            ]

        step2message = {0: list(range(len(messages)))}
        step2token = {0: {'output_token': 0, 'input_tokens': 0}}
        records = {'overall': {}, 'steps': []}
        finished = False
        cur_step_index = begin_step

        while cur_step_index < self.MAX_STEP + begin_step:
            cur_step_index += 1
            msg_start_idx = len(messages)
            use_token = {}
            use_time = 0.0
            assistant_msg = None
            observations = []

            try:
                assistant_msg, messages, use_token, use_time = self.act(messages)
                observations, finished = self.observe(assistant_msg, workspace, step=cur_step_index)
                messages.extend(observations)
            except Exception as e:
                error_phase = "observe" if assistant_msg is not None else "act"
                error_content = (
                    f"[Agent Error] An error occurred during '{error_phase}' at step {cur_step_index}: {e}\n"
                    "Please try a different approach or use the finish tool if you cannot proceed."
                )
                error_msg = {"role": "user", "content": error_content}
                messages.append(error_msg)
                observations = [error_msg]

            step2message[cur_step_index] = list(range(msg_start_idx, len(messages)))
            step2token[cur_step_index] = {
                'output_token': use_token.get('output_tokens', 0),
                'input_tokens': use_token.get('input_tokens', 0),
            }
            prev = cur_step_index - 1
            step2token[prev]['observation_token'] = (
                step2token[cur_step_index]['input_tokens']
                - step2token[prev]['input_tokens']
                - step2token[prev]['output_token']
            )
            self.save_current_states(cur_step_index, records, messages, use_token, use_time, assistant_msg, observations, step2message)

            if finished:
                break

        overall = {'total_steps': len(records['steps'])}
        numeric_keys = [
            'prompt_tokens', 'completion_tokens', 'reasoning_tokens',
            'cached_tokens', 'uncached_tokens', 'total_tokens',
            'accumulated_cost', 'input_tokens', 'output_tokens',
            'use_time',
        ]
        for key in numeric_keys:
            overall[key] = sum(
                step['metrics'].get(key, 0) for step in records['steps']
            )
        records['overall'] = overall
        records['step2message'] = step2message
        records['step2token'] = step2token
        with open(f"{self.output_dir}/records.json", "w") as f:
            json.dump(records, f, indent=4, default=str)

        save_jsonl(f"{self.output_dir}/messages.jsonl", messages)

        return messages, step2message
        

