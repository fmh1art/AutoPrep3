import json

from openhands.workspace import DockerWorkspace

from src.element.tools import *
from src.element.codegraph_tools import (
    CODEGRAPH_TOOLS,
    CODEGRAPH_PY_TOOLS,
    PYCODEGRAPH_TOOLS,
)


class CustomizedCodeAgentExecutor:

    def __init__(
        self,
        use_codegraph: bool = False,
        use_new_code_graph: bool = False,
        use_pycodegraph: bool = False,
    ):
        self._tools: dict[str, Tool] = {
            "terminal": ExecuteTerminal(),
            "file_editor": ExecuteFileEditor(),
            #TODO: 新增一个工具，用来分发agent来做一个子任务，这个tool有两个参数：1. 子任务描述（简洁）2. 解决这个子任务必须的observation对应的step indices，用来初始化agent的messages
            "finish": ExecuteFinish(),
        }
        if use_pycodegraph:
            # 优先使用 PyCodeGraph（零依赖，Python 专用，本地建图）
            for tool in PYCODEGRAPH_TOOLS:
                self._tools[tool.NAME] = tool
        elif use_new_code_graph:
            for tool in CODEGRAPH_PY_TOOLS:
                self._tools[tool.NAME] = tool
        elif use_codegraph:
            for tool in CODEGRAPH_TOOLS:
                self._tools[tool.NAME] = tool

    def get_tool_definitions(self) -> list[dict]:
        return [tool.tool_definition() for tool in self._tools.values()]

    def execute_tool(self, tool_name: str, args: dict, workspace: DockerWorkspace) -> str:
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self._tools.keys())}"
        try:
            return tool.execute(args, workspace)
        except TimeoutError as e:
            return f"TimeoutError: {e}"
        except (json.JSONDecodeError, TypeError) as e:
            return f"json.JSONDecodeError: Invalid arguments for tool '{tool_name}': {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def get_file_editor(self) -> ExecuteFileEditor:
        return self._tools["file_editor"]