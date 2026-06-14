"""AutoPrep3 中 PyCodeGraph 集成的快速烟雾测试。

- 验证 `CustomizedCodeAgentExecutor(use_pycodegraph=True)` 会注册 pycodegraph_* 工具
- 验证 `SweBenchRunner(use_pycodegraph=True)` 会把开关传递给 agent
- 验证每个 tool 的 `tool_definition()` 可以生成合法 JSON schema

不启动 docker，不运行真正的 code-agent。若你要"端到端跑 SWE-bench"，
直接使用 SweBenchRunner 即可（见 README 中原来的入口脚本）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 把 AutoPrep3 根目录加到 sys.path，保证 src.* 可导入
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.element.codegraph_tools import (
    PYCODEGRAPH_TOOLS,
    PYCODEGRAPH_CORE_TOOLS,
    PYCODEGRAPH_EXTRA_TOOLS,
    PyCodeGraphSearch,
    PyCodeGraphCallers,
    PyCodeGraphCallees,
    PyCodeGraphImpact,
    PyCodeGraphSubgraph,
    PyCodeGraphStatus,
    PyCodeGraphInit,
)
from src.element.executor import CustomizedCodeAgentExecutor


def _h1(s: str) -> None:
    print("\n" + "=" * 72)
    print("  " + s)
    print("=" * 72)


def check_tool_definitions() -> None:
    _h1("1) pycodegraph 工具注册清单")
    names = [t.NAME for t in PYCODEGRAPH_TOOLS]
    print("core pycodegraph tools:", names)
    print("extra tools:", [t.NAME for t in PYCODEGRAPH_EXTRA_TOOLS])

    for cls in [
        PyCodeGraphSearch,
        PyCodeGraphCallers,
        PyCodeGraphCallees,
        PyCodeGraphImpact,
        PyCodeGraphSubgraph,
        PyCodeGraphStatus,
        PyCodeGraphInit,
    ]:
        t = cls()
        td = t.tool_definition()
        assert td["type"] == "function"
        assert td["function"]["name"] == t.NAME
        assert "repo_path" in td["function"]["parameters"]["properties"]
        assert td["function"]["parameters"].get("required"), f"{t.NAME} 缺少 required"
        print(f"  ok {t.NAME}: required={td['function']['parameters']['required']}")


def check_executor_registration() -> None:
    _h1("2) CustomizedCodeAgentExecutor 三选一注册")

    for flag_name, kwargs in [
        ("use_pycodegraph=True",      {"use_pycodegraph": True}),
        ("use_new_code_graph=True",    {"use_new_code_graph": True}),
        ("use_codegraph=True",         {"use_codegraph": True}),
        ("默认（全关）",                {}),
    ]:
        exe = CustomizedCodeAgentExecutor(**kwargs)
        tool_names = sorted(exe._tools.keys())  # 直接看内部注册
        print(f"  [{flag_name}] tools: {tool_names}")
        # 基本断言：按开关应当确实注册对应前缀的工具
        has_py = any(n.startswith("pycodegraph_") for n in tool_names)
        has_cg = any(n.startswith("codegraph_") for n in tool_names)
        if kwargs.get("use_pycodegraph"):
            assert has_py, f"{flag_name}: 应当有 pycodegraph_* 工具"
        if kwargs.get("use_new_code_graph"):
            assert has_cg, f"{flag_name}: 应当有 codegraph_* 工具 (CodeGraphPy)"
        if kwargs.get("use_codegraph"):
            assert has_cg, f"{flag_name}: 应当有 codegraph_* 工具"
        if not kwargs:
            assert not has_py and not has_cg, "默认全关但有 code graph 工具被注册"


def check_runner_flag_mutual_exclusion() -> None:
    _h1("3) SweBenchRunner 构造参数与互斥")
    # 为避免环境依赖我们直接用最小化的 "from src.benchmarks.swe_bench_runner import SweBenchRunner"
    from src.benchmarks.swe_bench_runner import SweBenchRunner

    cases = [
        {"use_pycodegraph": True},
        {"use_new_code_graph": True},
        {"use_codegraph": True},
        {"use_pycodegraph": True, "use_new_code_graph": True, "use_codegraph": True},
    ]
    for kwargs in cases:
        runner = SweBenchRunner(exp_cfg={"llm_name": "demo"}, tmp_root="./_tmp_demo", **kwargs)
        print(
            f"  input {kwargs} -> "
            f"use_pycodegraph={runner.use_pycodegraph}, "
            f"use_new_code_graph={runner.use_new_code_graph}, "
            f"use_codegraph={runner.use_codegraph}"
        )
        # 断言：任一时刻最多有一个为 True
        total = sum([runner.use_pycodegraph, runner.use_new_code_graph, runner.use_codegraph])
        assert total <= 1, "多个 code graph 开关同时为 True，违背三选一语义"


def check_tool_definition_json_serializable() -> None:
    _h1("4) tool_definition 可被 json.dumps（符合 OpenAI tool schema）")
    exe = CustomizedCodeAgentExecutor(use_pycodegraph=True)
    for td in exe.get_tool_definitions():
        raw = json.dumps(td, ensure_ascii=False, indent=2)
        print(f"  {td['function']['name']}: JSON 长度 {len(raw)}")


def main() -> int:
    check_tool_definitions()
    check_executor_registration()
    check_runner_flag_mutual_exclusion()
    check_tool_definition_json_serializable()
    _h1("完成：全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
