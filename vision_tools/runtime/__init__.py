"""
StateManager re-export from runtime package.

Originally located at ``core/state_manager.py``, now re-exported here
for architectural consistency (runtime services belong in runtime/).
"""
from vision_tools.core.state_manager import (
    StateManager,
    StateRule,
    RuleType,
    ZoneDefinition,
    Alert,
)

__all__ = ["StateManager", "StateRule", "RuleType", "ZoneDefinition", "Alert"]
