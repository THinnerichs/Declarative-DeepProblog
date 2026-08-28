"""Load SLASH's MVPP under a distinct module name.

Both NeurASP and SLASH ship a module called `mvpp`; whichever is
imported first wins in sys.modules. This loader imports SLASH's copy
explicitly from its file path so the two can coexist in one process.
"""
import importlib.util
import sys
from pathlib import Path

_SLASH_SRC = Path(__file__).parent.parent.parent / "slash_examples" / "slash_src" / "SLASH"

_spec = importlib.util.spec_from_file_location("slash_mvpp",
                                               _SLASH_SRC / "mvpp.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["slash_mvpp"] = _mod
_spec.loader.exec_module(_mod)

SlashMVPP = _mod.MVPP
