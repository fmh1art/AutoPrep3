

import json
import os
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import all_filepaths_in_dir, open_json, save_json, load_jsonl, parse_any_string


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Agent cost three-dimension analysis.")
    parser.add_argument("--root", type=str,
                        default="_tmp/code_agent_limit64_doubao_2026-05-16_18-59-41")
    return parser.parse_args()

def query_llm_get_meta_info_one_data(messages, step2message, save_path):
    PROMPT_BEFORE = """You are a helpful assistant for analyzing the agent's behavior.
    
Given the conversation history between the user and the agent, you should generate a meta information for the last step. Here is the conversation history:"""

    PROMPT_FOCUS_ON_STEP = """Your task is to annotate the CURRENT step only, according to the META_INFO_PARAM schema defined above.

Return ONLY a valid JSON object that conforms exactly to the schema:
```json
{
  "intention": "Progressive" | "Refinement",
  "effection": "Read" | "Write" | "Meta",
  "related_step_indexs": [integer, ...]
}
```

Annotation rules:

1. Determine "intention" by causal dependency, not by surface wording.
   - Use "Refinement" only when the current step exists because a previous step failed, produced an unexpected result, or needs to be redone/modified.
   - Use "Progressive" when the current step advances to a new sub-goal that has not previously been attempted.
   - The first attempt at any sub-goal is always "Progressive", even if it later fails.

2. Determine "effection" by environmental effect.
   - Use "Read" if the step only observes or retrieves information, including running tests/scripts for diagnostic output.
   - Use "Write" if the step persistently changes files, packages, environment state, or other external state.
   - Use "Meta" if the step only summarizes, concludes, reports, or reasons without reading from or writing to the external environment.

3. Determine "related_step_indexs" carefully.
   - Use 1-based indexes of prior steps only.
   - Include only prior steps whose outputs or effects are necessary to reconstruct why the current step was taken and what it contains.
   - Do NOT include the current step.
   - Do NOT include the original task description itself.
   - Do NOT default to the immediately previous step unless its content is truly required.
   - For Refinement steps, include both:
     a) the prior step being repaired/redone/modified, and
     b) the prior step whose failure or unexpected result triggered the refinement.
   - For diagnostic Read steps, include the Write step(s) or action(s) being diagnosed.
   - For Meta/finish/summary steps, include the key prior steps whose results are being summarized.

4. Output constraints:
   - Output JSON only.
   - Do not include markdown fences.
   - Do not include explanations, comments, or extra fields.
   - The JSON must be syntactically valid.
   - The value of "intention" must be exactly one of: "Progressive", "Refinement".
   - The value of "effection" must be exactly one of: "Read", "Write", "Meta".
   - The value of "related_step_indexs" must be an array of integers.

Now analyze the current step and generate the meta information in the format of ```your_meta_info_here``` with NO extra text."""

    # finally return a dict: step2meta_info
    step2meta_info = {}
    tol_steps = sorted(list(step2message.keys()))

    llm = SimpleAPICaller(llm_name="ep-20250612104210-ss27q", api_key='02002c34-ca56-4797-a742-87fd6ec3981c', base_url='https://ark-cn-beijing.bytedance.net/api/v3', cache_server_url='http://127.0.0.1:18731')

    for step in tol_steps:
        msg_idxs = step2message[step]
        first_msg_idx = min(msg_idxs)
        last_msg_idx = max(msg_idxs)
        before_msgs = [dict(m, role='user') if m.get('role') == 'system' else dict(m) for m in messages[:first_msg_idx]]
        query_msgs = [{'role': 'system', 'content': PROMPT_BEFORE}] + before_msgs
        query_msgs.append({'role': 'user', 'content': f"Here is the step you need to generate meta information for:"})
        query_msgs = query_msgs + messages[first_msg_idx:last_msg_idx+1]
        query_msgs.append({'role': 'user', 'content': PROMPT_FOCUS_ON_STEP})
        
        meta_info = None
        for _ in range(5):
            try:
                response = llm.chat(messages=query_msgs)
                meta_info = parse_any_string(response, code_type='json', hard_replace='your_meta_info_here')
                # check the format of meta_info
                if 'intention' not in meta_info or 'effection' not in meta_info or 'related_step_indexs' not in meta_info:
                    query_msgs.append({'role': 'user', 'content': f"The JSON you returned is missing required fields (intention, effection, related_step_indexs). Please strictly follow the schema and return a valid JSON object with fields: intention, effection, related_step_indexs. Here is the JSON you returned: {response}"})
                    continue
                if meta_info['intention'] not in ['Progressive', 'Refinement']:
                    query_msgs.append({'role': 'user', 'content': f"The value of 'intention' must be exactly one of: 'Progressive', 'Refinement'. Please correct it and return a valid JSON object. Here is the JSON you returned: {response}"})
                    continue
                if meta_info['effection'] not in ['Read', 'Write', 'Meta']:
                    query_msgs.append({'role': 'user', 'content': f"The value of 'effection' must be exactly one of: 'Read', 'Write', 'Meta'. Please correct it and return a valid JSON object. Here is the JSON you returned: {response}"})
                    continue
                if not isinstance(meta_info['related_step_indexs'], list) or not all(isinstance(idx, int) for idx in meta_info['related_step_indexs']):
                    query_msgs.append({'role': 'user', 'content': f"The value of 'related_step_indexs' must be an array of integers. Please correct it and return a valid JSON object. Here is the JSON you returned: {response}"})
                    continue
            except Exception as e:
                query_msgs.append({'role': 'user', 'content': f"Error parsing your response as JSON: {str(e)}. Please return a valid JSON object that conforms to the schema. Here is the response you returned: {response}"})
                continue
            break
        
        step2meta_info[step] = meta_info
        print(f"[Step {step}] meta_info: {meta_info} for file {save_path}")
    
    save_json(step2meta_info, save_path)
    print(f"Saved meta information for all steps to {save_path}")
    
    return step2meta_info

def load_data(args):
    
    data = [] # load message, step2message obj
    for fn in all_filepaths_in_dir(os.path.join(args.root, "log"), endswith='records.json'):
        records = open_json(fn)
        step2message = {}
        for step in records['steps']:
            step_idx = step['step']
            message_indices = step['message_indices']
            step2message[step_idx] = message_indices
        
        messages = load_jsonl(fn.replace('records.json', 'messages.jsonl'))
        data.append((messages, step2message, fn.replace('records.json', 'step2meta_info.json')))
    return data


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(query_llm_get_meta_info_one_data, messages, step2message, save_path): save_path
                   for messages, step2message, save_path in data}
        for future in as_completed(futures):
            save_path = futures[future]
            try:
                future.result()
                print(f"[Done] {save_path}")
            except Exception as e:
                print(f"[Error] {save_path}: {e}")