"""Anti-cheat detection system."""

from referee.anticheat.base import CompositeAntiCheatChecker
from referee.anticheat.hardcoded_output import HardcodedOutputCheck
from referee.anticheat.shortcut_algo import ShortcutAlgorithmCheck
from referee.anticheat.env_snooping import EnvironmentSnoopingCheck

__all__ = [
    "CompositeAntiCheatChecker",
    "HardcodedOutputCheck",
    "ShortcutAlgorithmCheck",
    "EnvironmentSnoopingCheck",
]
