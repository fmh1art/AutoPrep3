import copy
import json

from openhands.workspace import DockerWorkspace
from src.element.tools import (
    Tool,
    ExecuteTerminal,
    ExecuteFileEditor,
    ExecuteFinish,
)

META_INFO_PARAM = {
    "type": "object",
    "description": (
        "Typological analysis of the current tool call. "
        "This meta information characterizes the intention, effect, "
        "and dependency relationship of this action within the overall task. "
        "IMPORTANT: Annotate based on OBJECTIVE signals (e.g., previous error messages, "
        "explicit references to prior content), NOT on surface-level verbs (e.g., "
        "'edit' does not automatically mean Refinement; 'install' does not automatically "
        "mean Progressive)."
    ),
    "properties": {
        "intention": {
            "type": "string",
            "enum": ["Progressive", "Refinement"],
            "description": (
                "Classify by whether this step exists BECAUSE a previous step failed "
                "or produced an unexpected result.\n"
                "\n"
                "* 'Refinement' — This step exists because one or more previous steps "
                "  did not achieve the intended effect. Concretely, mark as Refinement "
                "  if ANY of the following holds:\n"
                "    (a) An immediately preceding step returned a non-zero exit code, "
                "        an exception, or an error message, and the current step "
                "        attempts to recover from it (e.g., switching to a different "
                "        package name after a failed install, retrying with different "
                "        arguments, fixing a syntax error introduced earlier).\n"
                "    (b) The current step modifies, reverts, or re-issues an action "
                "        that was already performed in a prior step (e.g., re-editing "
                "        the same code region, re-running the same command with "
                "        adjusted parameters).\n"
                "    (c) The current step exists solely to compensate for a "
                "        previously unmet dependency or environmental gap discovered "
                "        through failure (e.g., installing a package after an "
                "        ImportError).\n"
                "\n"
                "* 'Progressive' — This step advances the task to a new sub-goal that "
                "  was not previously attempted, regardless of whether it succeeds. "
                "  The FIRST attempt at any sub-goal is Progressive, even if it later "
                "  turns out to be wrong. Examples:\n"
                "    - First reading of a file to understand the codebase.\n"
                "    - First edit that implements the intended fix.\n"
                "    - Creating a new test script for the first time.\n"
                "    - The first time installing a missing package after discovering "
                "      it is needed (this is Progressive, but the SECOND attempt with "
                "      a different package name is Refinement).\n"
                "\n"
                "Decision rule: ask 'Would this step exist if all previous steps had "
                "succeeded perfectly?' If NO -> Refinement. If YES -> Progressive."
            ),
        },
        "effection": {
            "type": "string",
            "enum": ["Read", "Write", "Meta"],
            "description": (
                "Classify by whether this step changes the external environment.\n"
                "\n"
                "* 'Read' — Retrieves or observes information without modifying the "
                "  environment. Examples: viewing files, listing directories, running "
                "  read-only commands (find, grep, cat, ls), executing test scripts to "
                "  observe their output (the script itself runs, but the intent is to "
                "  read the result), printing variables.\n"
                "  Note: Running a test or a Python script is 'Read' if its purpose is "
                "  to observe output for diagnosis, even though it technically executes "
                "  code.\n"
                "\n"
                "* 'Write' — Persistently changes the environment state. Examples: "
                "  editing files, creating files, deleting files, installing or "
                "  uninstalling packages, building extensions, modifying environment "
                "  variables, applying patches.\n"
                "\n"
                "* 'Meta' — The step neither reads from nor writes to the external "
                "  environment. It only reports, concludes, or summarizes the agent's "
                "  internal state. Examples: 'finish' tool calls, final summary "
                "  messages, reasoning-only steps that produce no environmental "
                "  side-effect."
            ),
        },
        "related_step_indexs": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "List of 1-based step indexes whose information is NECESSARY for "
                "reproducing the current step's decision and content. The current "
                "step, given only (i) the original task description and (ii) the "
                "outputs of the listed prior steps, should be reconstructable.\n"
                "\n"
                "CRITICAL GUIDELINES — read carefully:\n"
                "\n"
                "1. DO NOT default to [t-1]. The previous step is NOT automatically "
                "   a dependency. Only include it if its specific content is actually "
                "   used by the current step.\n"
                "\n"
                "2. INCLUDE ALL TRUE SOURCES, even if distant. A step may depend on "
                "   information from many steps ago. For example, a code edit at step "
                "   10 may depend on a file view at step 2 (to know what to edit), "
                "   even though steps 3-9 came in between.\n"
                "\n"
                "3. FOR REFINEMENT STEPS: include BOTH (a) the step being repaired/"
                "   redone (the 'target') AND (b) the step whose failure triggered "
                "   this refinement (the 'trigger'). These are often different steps.\n"
                "   Example: step 5 edits code; step 6 runs test; step 6 fails; "
                "   step 7 re-edits code -> step 7's deps = [5, 6].\n"
                "\n"
                "4. FOR DIAGNOSTIC READS (e.g., running tests after a fix): include "
                "   the step(s) whose effect is being diagnosed (e.g., the fix step), "
                "   not just the immediately preceding setup step.\n"
                "\n"
                "5. FOR FINISH/SUMMARY STEPS (effection='Meta'): include the key "
                "   steps whose results are being reported (typically the main "
                "   Progressive Write steps that accomplished the task), NOT an empty "
                "   list.\n"
                "\n"
                "6. EXCLUDE merely co-occurring steps. If a prior step happened "
                "   chronologically but its content is not used, do NOT include it.\n"
                "\n"
                "7. EXCLUDE the original task description itself. The task is "
                "   implicitly available; only list step indexes here.\n"
                "\n"
                "Examples:\n"
                "  - Step 2 views file X; step 5 edits a specific line in X based on "
                "    what was seen at step 2 -> deps = [2] (NOT [4]).\n"
                "  - Step 3 installs pkg-A and fails; step 4 installs pkg-B as "
                "    fallback -> deps = [3] (the failure that triggered the change).\n"
                "  - Step 23 is 'finish' summarizing the fix made at step 5 -> "
                "    deps = [5] (NOT [])."
            ),
        },
    },
    "required": ["intention", "effection", "related_step_indexs"],
}

def _inject_meta_info(tool_def: dict) -> dict:
    result = copy.deepcopy(tool_def)
    params = result["function"]["parameters"]
    params["properties"]["meta_info"] = META_INFO_PARAM
    if "required" in params:
        params["required"].append("meta_info")
    else:
        params["required"] = ["meta_info"]
    return result


class PreExpExecuteTerminal(ExecuteTerminal):

    def tool_definition(self):
        return _inject_meta_info(super().tool_definition())


class PreExpExecuteFileEditor(ExecuteFileEditor):

    def tool_definition(self):
        return _inject_meta_info(super().tool_definition())


class PreExpExecuteFinish(ExecuteFinish):

    def tool_definition(self):
        return _inject_meta_info(super().tool_definition())


class PreExpCodeAgentExecutor:

    def __init__(self):
        self._tools: dict[str, Tool] = {
            "terminal": PreExpExecuteTerminal(),
            "file_editor": PreExpExecuteFileEditor(),
            "finish": PreExpExecuteFinish(),
        }

    def get_tool_definitions(self) -> list[dict]:
        return [tool.tool_definition() for tool in self._tools.values()]

    def execute_tool(
        self, tool_name: str, args: dict, workspace: DockerWorkspace
    ) -> tuple[str, dict | None]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return (
                f"Error: Unknown tool '{tool_name}'. "
                f"Available tools: {list(self._tools.keys())}",
                None,
            )

        exec_args = copy.deepcopy(args)
        meta_info = exec_args.pop("meta_info", None)

        try:
            observation = tool.execute(exec_args, workspace)
            return observation, meta_info
        except TimeoutError as e:
            return f"TimeoutError: {e}", meta_info
        except (json.JSONDecodeError, TypeError) as e:
            return (
                f"json.JSONDecodeError: Invalid arguments for tool '{tool_name}': {e}",
                meta_info,
            )
        except Exception as e:
            return f"Error executing {tool_name}: {e}", meta_info
