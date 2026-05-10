"""Holder package for standalone OpenHands SDK tools that must be importable
from BOTH the host (client side) and inside the agent_server Docker container
under the *exact same* top-level qualname.

Importing this package adds its directory to ``sys.path`` so the standalone
modules inside (e.g. ``_oh_meta_subtask_finish_tool``) can be imported as
top-level modules — which matches how the same files are dropped into the
container's ``site-packages`` at sub-agent startup.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)

# Trigger registration of the SubtaskFinishTool with the SDK tool registry,
# and record the qualname (``_oh_meta_subtask_finish_tool``) that the
# RemoteConversation will forward to the agent_server for dynamic import.
import _oh_meta_subtask_finish_tool  # noqa: F401,E402

REMOTE_TOOLS_DIR = _THIS_DIR
SUBTASK_FINISH_TOOL_MODULE = "_oh_meta_subtask_finish_tool"
SUBTASK_FINISH_TOOL_FILE = _os.path.join(_THIS_DIR, "_oh_meta_subtask_finish_tool.py")
