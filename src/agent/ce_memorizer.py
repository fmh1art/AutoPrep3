import json
import os
from datetime import datetime

from src.tools.funcs import load_jsonl, save_jsonl, render_j2, parse_any_string
from src.module import SimpleAPICaller


def _normalize_fluctuation_ratio(indicators: dict) -> None:
    ratio = indicators.get('fluctuation_ratio')
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        raise ValueError(f"fluctuation_ratio must be a list of two numbers, got: {ratio}")
    values = []
    for v in ratio:
        v = float(v)
        if v > 2.0:
            v = v / 100.0
        v = max(0.0, min(2.0, v))
        values.append(v)
    a, b = values[0], values[1]
    if a > b:
        a, b = b, a
    indicators['fluctuation_ratio'] = [round(a, 4), round(b, 4)]


class CEMemorizer:
    """Memory的格式是：
    
    [
        {metadata: {包括这条memory的一些meta信息，如tool类型，llm_backbone名字等具有表示性的属性}, memory: {}},
        {}
    ]

    """

    def __init__(self, cfg: dict, memory_root:str='_tmp/memory'):
        self.llm = SimpleAPICaller(llm_name = cfg['llm_name'], api_key = cfg['key'], base_url = cfg['openai_base_url'], api_version = cfg.get('api_version', None))
        self.memory_root = memory_root
        self.task_memory, self.envirment_memory, self.backbone_memory = [], [], []
        self.initialize_memory()

    def initialize_memory(self):
        task_mem_path, env_mem_path, backbone_mem_path = os.path.join(self.memory_root, 'task_memory.jsonl'), os.path.join(self.memory_root, 'envirment_memory.jsonl'), os.path.join(self.memory_root, 'backbone_memory.jsonl')
        if os.path.exists(task_mem_path):
            self.task_memory = load_jsonl(task_mem_path)
        if os.path.exists(env_mem_path):
            self.envirment_memory = load_jsonl(env_mem_path)
        if os.path.exists(backbone_mem_path):
            self.backbone_memory = load_jsonl(backbone_mem_path)

    def summarize_task_memory(self, instruction: str, plan: list, trajectory: list, subtask_index: int, total_steps: int, taken_tool_list: list, output_token_list: list, observation_token_list: list, last_error:str):
        token_metrics = {
            'input_token': -1, 
            'output_token': -1,
            'cached_token': -1
        }

        prompt = render_j2('ce_memorize_subtask.j2', context={
            "instruction": instruction,
            "plan": plan,
            "subtask_index": subtask_index,
            "total_steps": total_steps,
            "trajectory": trajectory,
            "taken_tool_list": taken_tool_list,
            "output_token_list": output_token_list,
            "observation_token_list": observation_token_list,
        })

        if last_error:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]', f"The last error encountered is: {last_error}")
        else:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

        answer = self.llm.chat(prompt)
        try:
            parsed_answer = parse_any_string(answer, code_type='json')
            answer_dict = json.loads(parsed_answer)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM answer as JSON. Answer: {answer}. Error: {e}")

        def check_parsed_answer_structure(answer_dict):
            if not isinstance(answer_dict, dict):
                raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
            if 'subtask_type' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'subtask_type' key: {answer_dict}")
            if 'knowledge' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'knowledge' key: {answer_dict}")
            if 'indicators' not in answer_dict:
                raise ValueError(f"Parsed answer does not contain 'indicators' key: {answer_dict}")
            if 'uncertainty_factor_exists' not in answer_dict['indicators']:
                raise ValueError(f"Parsed answer's indicators do not contain 'uncertainty_factor_exists' key: {answer_dict}")
            if answer_dict['indicators']['uncertainty_factor_exists'] is True and 'fluctuation_ratio' not in answer_dict['indicators']:
                raise ValueError(f"Parsed answer's indicators indicate uncertainty factor exists but does not contain 'fluctuation_ratio' key: {answer_dict}")
            if answer_dict['indicators'].get('uncertainty_factor_exists') is True:
                _normalize_fluctuation_ratio(answer_dict['indicators'])
            uncertainty_val = answer_dict['indicators'].get('uncertainty')
            if uncertainty_val is not None:
                uncertainty_val = float(uncertainty_val)
                uncertainty_val = max(0.0, min(1.0, uncertainty_val))
                answer_dict['indicators']['uncertainty'] = round(uncertainty_val, 4)
            else:
                answer_dict['indicators']['uncertainty'] = 0.5 if answer_dict['indicators'].get('uncertainty_factor_exists') else 0.2

        check_parsed_answer_structure(answer_dict)
        parsed_answer = json.dumps(answer_dict, ensure_ascii=False)

        metric = self.llm.get_last_usage()
            
        token_metrics['input_token'] = metric.get('input_tokens', 0)
        token_metrics['output_token'] = metric.get('output_tokens', 0)
        token_metrics['cached_token'] = metric.get('cached_tokens', 0)

        return parsed_answer, token_metrics

    def summarize_llm_backbone(self, instruction: str, plan: str, llm_backbone_name: str, trajectory: list, last_error: str = None):
        token_metrics = {
            'input_token': -1,
            'output_token': -1,
            'cached_token': -1
        }

        prompt = render_j2('ce_memorize_backbone.j2', context={
            "instruction": instruction,
            "plan": plan,
            "llm_backbone_name": llm_backbone_name,
            "trajectory": trajectory,
        })

        if last_error:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]', f"The last error encountered is: {last_error}")
        else:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

        answer = self.llm.chat(prompt)
        try:
            parsed_answer = parse_any_string(answer, code_type='json')
            answer_dict = json.loads(parsed_answer)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM answer as JSON. Answer: {answer}. Error: {e}")

        if not isinstance(answer_dict, dict):
            raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
        if 'summary' not in answer_dict:
            raise ValueError(f"Parsed answer does not contain 'summary' key: {answer_dict}")

        metric = self.llm.get_last_usage()

        token_metrics['input_token'] = metric.get('input_tokens', 0)
        token_metrics['output_token'] = metric.get('output_tokens', 0)
        token_metrics['cached_token'] = metric.get('cached_tokens', 0)

        return parsed_answer, token_metrics
    
    def summarize_tool_observation(self, tool_name: str, tool_input: str, tool_observation: str, observation_token_count: int, last_error: str = None):
        token_metrics = {
            'input_token': -1,
            'output_token': -1,
            'cached_token': -1
        }

        prompt = render_j2('ce_memorize_tools.j2', context={
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_observation": tool_observation,
            "observation_token_count": observation_token_count,
        })

        if last_error:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]', f"The last error encountered is: {last_error}")
        else:
            prompt = prompt.replace('[LAST_ERROR_PLACEHOLDER]\n\n', "")

        answer = self.llm.chat(prompt)
        try:
            parsed_answer = parse_any_string(answer, code_type='json')
            answer_dict = json.loads(parsed_answer)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM answer as JSON. Answer: {answer}. Error: {e}")

        if not isinstance(answer_dict, dict):
            raise ValueError(f"Parsed answer is not a dictionary. Parsed answer: {answer_dict}")
        if 'summary' not in answer_dict:
            raise ValueError(f"Parsed answer does not contain 'summary' key: {answer_dict}")

        metric = self.llm.get_last_usage()

        token_metrics['input_token'] = metric.get('input_tokens', 0)
        token_metrics['output_token'] = metric.get('output_tokens', 0)
        token_metrics['cached_token'] = metric.get('cached_tokens', 0)

        return parsed_answer, token_metrics

    # ---- Memory formatting ----

    MEMORY_TYPE_MAP = {
        'task': 'task_memory',
        'backbone': 'backbone_memory',
        'environment': 'envirment_memory',
    }

    def _build_memory_entry(self, memory_type: str, summary_json: str, **extra_metadata) -> dict:
        """Convert a summarize result (JSON string) into the standard memory format:
        {"metadata": {...}, "memory": {...}}
        """
        memory_content = json.loads(summary_json) if isinstance(summary_json, str) else summary_json
        metadata = {
            "type": memory_type,
            "created_at": datetime.now().isoformat(),
        }
        metadata.update(extra_metadata)
        return {"metadata": metadata, "memory": memory_content}

    # ---- LLM-based memory admission ----

    def _judge_memory_admission(self, memory_type: str, candidate_entry: dict) -> dict:
        """Ask LLM whether a candidate memory should be added, rejected, or replace an existing one.
        Returns: {"decision": "accept"|"reject"|"replace", "reason": str, "replace_index": int}
        """
        attr_name = self.MEMORY_TYPE_MAP[memory_type]
        existing_memories = getattr(self, attr_name)

        existing_memories_str = json.dumps(
            [{"index": i, "memory": m.get("memory", m)} for i, m in enumerate(existing_memories)],
            ensure_ascii=False, indent=2
        ) if existing_memories else "[]"

        prompt = render_j2('ce_memory_judge.j2', context={
            "memory_type": memory_type,
            "candidate_content": json.dumps(candidate_entry.get("memory", candidate_entry), ensure_ascii=False, indent=2),
            "existing_count": len(existing_memories),
            "existing_memories": existing_memories_str,
        })

        answer = self.llm.chat(prompt)
        try:
            parsed_answer = parse_any_string(answer, code_type='json')
            judge_result = json.loads(parsed_answer)
        except Exception as e:
            raise ValueError(f"Failed to parse memory judge answer as JSON. Answer: {answer}. Error: {e}")

        if judge_result.get("decision") not in ("accept", "reject", "replace"):
            raise ValueError(f"Invalid decision in judge result: {judge_result}")
        return judge_result

    # ---- Memory add / save / serialize ----

    def add_memory(self, memory_type: str, summary_json: str, use_judge: bool = True, **extra_metadata) -> dict:
        """Build a memory entry from summarize output, optionally judge admission, and add to memory bank.

        Args:
            memory_type: one of 'task', 'backbone', 'environment'
            summary_json: the JSON string returned by a summarize_* method
            use_judge: if True, call LLM to decide whether to accept/reject/replace
            **extra_metadata: additional metadata fields (e.g. llm_backbone_name, tool_name)

        Returns:
            dict with keys: "added" (bool), "reason" (str), "token_metrics" (dict or None)
        """
        entry = self._build_memory_entry(memory_type, summary_json, **extra_metadata)
        attr_name = self.MEMORY_TYPE_MAP[memory_type]
        memory_bank = getattr(self, attr_name)

        token_metrics = None
        if use_judge and len(memory_bank) > 0:
            judge_result = self._judge_memory_admission(memory_type, entry)
            metric = self.llm.get_last_usage()
            token_metrics = {
                'input_token': metric.get('input_tokens', 0),
                'output_token': metric.get('output_tokens', 0),
                'cached_token': metric.get('cached_tokens', 0),
            }

            decision = judge_result["decision"]
            reason = judge_result.get("reason", "")

            if decision == "reject":
                return {"added": False, "reason": reason, "token_metrics": token_metrics}
            elif decision == "replace":
                replace_idx = judge_result.get("replace_index", -1)
                if 0 <= replace_idx < len(memory_bank):
                    memory_bank[replace_idx] = entry
                    return {"added": True, "reason": f"replaced index {replace_idx}: {reason}", "token_metrics": token_metrics}
                else:
                    memory_bank.append(entry)
                    return {"added": True, "reason": f"replace_index invalid, appended instead: {reason}", "token_metrics": token_metrics}

        memory_bank.append(entry)
        self.save_memory(memory_type)
        return {"added": True, "reason": "accepted", "token_metrics": token_metrics}

    def save_memory(self, memory_type: str = None):
        """Persist memory bank(s) to disk as JSONL files.
        If memory_type is None, save all three types.
        """
        type_to_file = {
            'task': 'task_memory.jsonl',
            'backbone': 'backbone_memory.jsonl',
            'environment': 'envirment_memory.jsonl',
        }
        types_to_save = [memory_type] if memory_type else list(type_to_file.keys())

        for mt in types_to_save:
            attr_name = self.MEMORY_TYPE_MAP[mt]
            data = getattr(self, attr_name)
            path = os.path.join(self.memory_root, type_to_file[mt])
            save_jsonl(path, data)

    def serialize_memory(self, memory_type: str = None, max_entries: int = None) -> str:
        """Serialize memory bank(s) into a human-readable text block for prompt injection.

        Args:
            memory_type: if specified, only serialize that type; otherwise serialize all
            max_entries: if specified, only include the most recent N entries per type

        Returns:
            A formatted string representation of the memory bank(s).
        """
        type_to_label = {
            'task': 'Task Memory',
            'backbone': 'LLM Backbone Memory',
            'environment': 'Environment / Tool Memory',
        }
        types_to_serialize = [memory_type] if memory_type else list(type_to_label.keys())
        sections = []

        for mt in types_to_serialize:
            attr_name = self.MEMORY_TYPE_MAP[mt]
            entries = getattr(self, attr_name)
            if max_entries is not None:
                entries = entries[-max_entries:]
            if not entries:
                continue

            lines = [f"## {type_to_label[mt]} ({len(entries)} entries)"]
            for i, entry in enumerate(entries):
                memory_content = entry.get("memory", entry)
                if isinstance(memory_content, dict):
                    content_str = json.dumps(memory_content, ensure_ascii=False, indent=2)
                else:
                    content_str = str(memory_content)
                lines.append(f"\n### Entry {i + 1}")
                lines.append(content_str)
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No memory entries available."

