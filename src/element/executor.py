import json

from openhands.workspace import DockerWorkspace

from src.element.codegraph_tools import PYCODEGRAPH_TOOLS
from src.element.tools import ExecuteFileEditor, ExecuteFinish, ExecuteTerminal, Tool


class CustomizedCodeAgentExecutor:

    def __init__(
        self,
        use_pycodegraph: bool = False,
    ):
        self._tools: dict[str, Tool] = {
            "terminal": ExecuteTerminal(),
            "file_editor": ExecuteFileEditor(),
            "finish": ExecuteFinish(),
        }
        if use_pycodegraph:
            # NOTE: PYCODEGRAPH_TOOLS is intentionally pruned in
            # src.element.codegraph_tools to avoid exposing low-yield interfaces
            # (for example: pycodegraph_callees / pycodegraph_impact /
            # pycodegraph_subgraph / pycodegraph_status / pycodegraph_init)
            # in the initial tool schema.
            for tool in PYCODEGRAPH_TOOLS:
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
