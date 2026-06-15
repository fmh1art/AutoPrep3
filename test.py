input_token_price = 1
output_token_price = 2
cached_token_price = 0.02

first_prompt_tok = 100
turns = [
    # output, obs
    (10, 100),
    (10, 100),
    (10, 100),
    (10, 100)
]

# 计算考虑cache，同时多轮prompt的成本
def calculate_cost_with_cache_multiturn(
    input_token_price,
    output_token_price,
    cached_token_price,
    first_prompt_tok,
    turns
):
    total_cost = 0
    # 上一轮已经进入 prompt、可被 cache 的 token
    cached_tokens = 0

    for i, (output_tok, obs_tok) in enumerate(turns):
        if i == 0:
            # 第 1 轮：初始 prompt 是新的 uncached input
            uncached_input_tokens = first_prompt_tok
            cached_input_tokens = 0
        else:
            # 第 2 轮开始：
            # 之前已经出现过的 prompt 可走 cache
            cached_input_tokens = cached_tokens

            # 当前新加入的是上一轮产生的 obs
            uncached_input_tokens = turns[i - 1][1]

        output_tokens = output_tok

        turn_cost = (
            cached_input_tokens * cached_token_price
            + uncached_input_tokens * input_token_price
            + output_tokens * output_token_price
        )

        total_cost += turn_cost

        # 当前轮结束后，下一轮可 cache 的上下文包括：
        # 已有 cached 内容 + 本轮 uncached input + 本轮 output
        cached_tokens = (
            cached_input_tokens
            + uncached_input_tokens
            + output_tokens
        )

    return total_cost

def calculate_cost_with_cache_singleturn(
    input_token_price,
    output_token_price,
    first_prompt_tok,
    turns
):
    # 直接计算最后一轮的成本，假设前面的内容直接给出。
    # 例如：
    # input = first_prompt_tok + sum(前面所有轮的 output + obs)
    # output = 最后一轮 output

    if not turns:
        return first_prompt_tok * input_token_price

    previous_context_tokens = 0

    # 最后一轮之前的所有 output 和 obs 都作为输入上下文
    for output_tok, obs_tok in turns[:-1]:
        previous_context_tokens += output_tok + obs_tok

    input_tokens = first_prompt_tok + previous_context_tokens
    output_tokens = turns[-1][0]

    total_cost = (
        input_tokens * input_token_price
        + output_tokens * output_token_price
    )

    return total_cost

cost_with_cache_multiturn = calculate_cost_with_cache_multiturn(
    input_token_price,
    output_token_price,
    cached_token_price,
    first_prompt_tok,
    turns
)

cost_with_cache_singleturn = calculate_cost_with_cache_singleturn(
    input_token_price,
    output_token_price,
    first_prompt_tok,   
    turns
)

print(f"Cost with cache (multi-turn): {cost_with_cache_multiturn}")
print(f"Cost with cache (single-turn): {cost_with_cache_singleturn}")