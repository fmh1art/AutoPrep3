import json, copy, os
import logging
import time

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import compute_metrics, write_messages_to_markdown_format, setup_file_logger, save_jsonl
from src.element.executor import CustomizedCodeAgentExecutor

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_STEP = 5

SYSTEM_PROMPT = """You are an agent, a helpful AI assistant that use pre-defined tools to solve tasks.""".strip()

CODEGRAPH_PROMPT = """
This workspace has been pre-indexed with **CodeGraph**, a local code-intelligence
knowledge graph (symbols, callers, imports, framework routes) stored at
`<repo_path>/.codegraph/`. You have 3 dedicated tools: `codegraph_search`,
`codegraph_context`, and `codegraph_callers`. Additional sub-commands (callees,
impact, files, sync, affected) are available via the terminal tool, e.g.
`codegraph impact <symbol> --path <repo_path>`.

**CRITICAL RULES — violating these wastes tokens:**

1. **Use `codegraph_search` INSTEAD of `grep -r`** to find symbol definitions.
   Example: `codegraph_search(query="URLValidator", repo_path="/workspace/django")`
2. **Use `codegraph_context` ONLY with a SHORT, keyword-focused query** (max 120
   chars). Do NOT paste the entire issue description — extract the key class/function
   names and the core problem. Example: `codegraph_context(task="URLValidator
   username/password validation RFC 1738", repo_path="/workspace/django")`
3. **Use `codegraph_callers` to find usages** instead of grepping for a symbol name.
4. **NEVER re-verify CodeGraph results with grep/find**. Treat returned file paths
   and line numbers as authoritative — go straight to `file_editor view` on the
   specific file:line if you need to see the code.
5. **Do NOT call `codegraph_context` and then immediately `codegraph_search` for
   the same symbol** — that's redundant. Pick one.
6. After editing files, you do NOT need to call `codegraph sync` — the index
   auto-syncs in the background.

All `codegraph_*` tools require `repo_path` — the absolute path of the repo root
inside the workspace (provided in the task prompt).
""".strip()

CODEGRAPH_PY_PROMPT = """
This workspace has been pre-indexed with **CodeGraphPy**, a Python-optimized
local code-intelligence knowledge graph (symbols, callers, imports, framework routes)
stored at `<repo_path>/.codegraph_py/`. It provides PRECISE location tracking including
exact line ranges (start_line, end_line) and column positions for every symbol.

You have 3 dedicated tools: `codegraph_search`, `codegraph_context`, and `codegraph_callers`.
Additional tools: `codegraph_callees`, `codegraph_impact`, `codegraph_status`,
`codegraph_sync`, `codegraph_search_by_location`, `codegraph_search_routes`.

**ENHANCED FEATURES — CodeGraphPy provides:**

- **Precise location tracking**: Every result includes `start_line`, `end_line`,
  `start_column`, `end_column` — you know exactly where each symbol is defined.
- **Type information**: Returns type annotations and inferred types for variables,
  parameters, and return values.
- **Definition code**: Returns the actual source code of each symbol's definition.
- **Python-optimized**: Better handling of Python imports, decorators, type hints,
  and framework patterns (Flask, FastAPI, Django).
- **Field-qualified queries**: Use `kind:function calculate` or `path:utils helper`
  to narrow searches.
- **Route search**: `codegraph_search_routes` finds web framework route definitions.
- **Location search**: `codegraph_search_by_location` finds the symbol at a file:line.

**CRITICAL RULES — violating these wastes tokens:**

1. **Use `codegraph_search` INSTEAD of `grep -r`** to find symbol definitions.
   Example: `codegraph_search(query="URLValidator", repo_path="/workspace/django")`
   Example with filter: `codegraph_search(query="kind:function calculate", repo_path="/workspace/django")`
2. **Use `codegraph_context` ONLY with a SHORT, keyword-focused query** (max 120
   chars). Do NOT paste the entire issue description — extract the key class/function
   names and the core problem. Example: `codegraph_context(task="URLValidator
   username/password validation RFC 1738", repo_path="/workspace/django")`
3. **Use `codegraph_callers` to find usages** instead of grepping for a symbol name.
4. **NEVER re-verify CodeGraphPy results with grep/find**. Treat returned file paths
   and line ranges as authoritative — go straight to `file_editor view` on the
   specific file:line if you need to see the code.
5. **Do NOT call `codegraph_context` and then immediately `codegraph_search` for
   the same symbol** — that's redundant. Pick one.
6. After editing files, call `codegraph_sync` to update the index for changed files.
7. **Use `codegraph_search_by_location`** when you have a file:line reference and
   want to know what symbol is there.
8. **Use `codegraph_search_routes`** to find web framework route definitions.

All `codegraph_*` tools require `repo_path` — the absolute path of the repo root
inside the workspace (provided in the task prompt).
""".strip()

PYCODEGRAPH_PROMPT = """
This workspace has been pre-indexed with **PyCodeGraph**, a lightweight, zero-dependency
local code-intelligence knowledge graph (Python symbols, references, containment)
stored at `<repo_path>/.pycodegraph/graph.json`.

You have the following dedicated tools (all require `repo_path`):

- `pycodegraph_search(query)` — find symbol definitions by name/node_id. Returns
  file, start_line, end_line, kind, and stable id.
- `pycodegraph_callers(symbol)` — list symbols that REFERENCE the given symbol
  (what calls/uses it). Useful before editing.
- `pycodegraph_callees(symbol)` — list symbols REFERENCED BY the given symbol
  (what it depends on).
- `pycodegraph_impact(symbol, depth)` — BFS over references to get the change
  radius of a symbol; gives you the set of potentially affected callers.
- `pycodegraph_subgraph(prefix)` — cut a sub-graph by file-path prefix, so you
  can focus on one package/module at a time.
- `pycodegraph_status` — index size summary (nodes, edges, kind distribution).
- `pycodegraph_init` — (re)build the index. Normally automatic; use only when
  status reports 0 nodes.

All tools return structured JSON so you can directly pick file/line numbers.

**CRITICAL RULES for PyCodeGraph:**

1. Use `pycodegraph_search` INSTEAD of `grep -r` for locating definitions.
2. Use `pycodegraph_callers` / `pycodegraph_impact` to decide how wide a change
   really is before editing — this avoids breaking distant callers.
3. When looking at a file, `pycodegraph_subgraph` gives you the module-local
   symbol map, cheap.
4. Never re-verify results with grep/find — treat returned file paths and
   start_line/end_line as authoritative.
5. repo_path is the absolute path of the repo root inside the container, e.g.
   `/workspace/<repo_name>` (provided in the task prompt).
""".strip()


TASK_PROMPT = """
You need to solve the following task:

{task}
""".strip()

SKILL_PROMPT = """
You have the following skills:

<meta_skill>
{meta_skill}
</meta_skill>

<tool_skill>
{tool_skill}
</tool_skill>
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


from openhands.workspace import DockerWorkspace

class CustomizedCodeAgent:
    
    def __init__(self, llm_cfg: dict, output_dir: str, max_step:int=200, use_skill: bool = True, skill_path: str | None = None, use_codegraph: bool = False, use_new_code_graph: bool = False, use_pycodegraph: bool = False):
        self.llm = SimpleAPICaller(
            llm_name=llm_cfg.get('llm_name', None),
            api_key=llm_cfg.get('key', None),
            base_url=llm_cfg.get('openai_base_url', None),
        )
        self.MAX_STEP = max_step
        self.MAX_RETRY_LLM_CALL = 3
        self.output_dir = output_dir
        self.use_codegraph = use_codegraph
        self.use_new_code_graph = use_new_code_graph
        self.use_pycodegraph = use_pycodegraph
        self.executor = CustomizedCodeAgentExecutor(
            use_codegraph=use_codegraph,
            use_new_code_graph=use_new_code_graph,
            use_pycodegraph=use_pycodegraph,
        )
        self.tools = self.executor.get_tool_definitions()
        self._fake_response_count = 0
        self._max_fake_responses = 10
        self.skill_path = skill_path
        self.use_skill = use_skill if skill_path is None else True
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
        with open(f"{self.output_dir}/log/step_{step}.md", "w") as f:
            f.write(record_md)
        with open(f"{self.output_dir}/records.json", "w") as f:
            json.dump(records, f, indent=4, default=str)
        
    def _install_skill(self, workspace: DockerWorkspace):
        skill_path = self.skill_path or '_tmp/skill_evolver_doubao_2026-05-20_22-25-36/.skill'
        if os.path.isdir(skill_path):
            import tarfile, io, base64
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w:gz') as tar:
                tar.add(skill_path, arcname='.skill')
            encoded = base64.b64encode(buf.getvalue()).decode()
            workspace.execute_command(
                f"cd /workspace && echo '{encoded}' | base64 -d | tar -xzf -",
                timeout=120,
            )

        meta_skill_str = ""
        meta_skill_file = os.path.join(skill_path, 'meta_skill', 'meta_skill.md')
        if os.path.isfile(meta_skill_file):
            with open(meta_skill_file, 'r') as f:
                meta_skill_str = f.read()

        tool_skill_dir = os.path.join(skill_path, 'tool_skill')
        tool_skill_names = []
        tool_skill_descriptions = {}
        if os.path.isdir(tool_skill_dir):
            for name in sorted(os.listdir(tool_skill_dir)):
                skill_md = os.path.join(tool_skill_dir, name, 'skill.md')
                if os.path.isfile(skill_md):
                    tool_skill_names.append(name)
                    with open(skill_md, 'r') as f:
                        tool_skill_descriptions[name] = f.read()

        if tool_skill_names:
            names_str = ', '.join(tool_skill_names)
            desc_parts = []
            for name in tool_skill_names:
                desc_parts.append(f"The description for {name} is <{name}> {tool_skill_descriptions[name]} </{name}> To use this skill, you can run `bash /workspace/.skill/{name}/skill.sh [args]`")
            tool_skill_str = f"There are {len(tool_skill_names)} tool skills, named {names_str}. " + ' '.join(desc_parts)
        else:
            tool_skill_str = ""

        system_prompt = SYSTEM_PROMPT + "\n" + SKILL_PROMPT.format(meta_skill=meta_skill_str, tool_skill=tool_skill_str)
        return system_prompt
        

    def run(self, task_instruction: str, workspace: DockerWorkspace, messages: list[dict]=None, begin_step: int = 0):
        self._fake_response_count = 0
        system_prompt = SYSTEM_PROMPT
        if self.use_skill:
            system_prompt = self._install_skill(workspace)
        if self.use_pycodegraph:
            system_prompt = system_prompt + "\n\n" + PYCODEGRAPH_PROMPT
        elif self.use_new_code_graph:
            system_prompt = system_prompt + "\n\n" + CODEGRAPH_PY_PROMPT
        elif self.use_codegraph:
            system_prompt = system_prompt + "\n\n" + CODEGRAPH_PROMPT

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
        