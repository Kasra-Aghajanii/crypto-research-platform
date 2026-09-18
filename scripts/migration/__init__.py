"""Database migrations.

Migration modules are named ``NNNN_description.py`` so that ordering is obvious
on disk.  Those names are not valid Python identifiers, so they are loaded by
path here and re-exported under importable aliases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MIGRATIONS_DIR = Path(__file__).parent


def load_migration(filename: str) -> ModuleType:
    """Load a numerically-named migration module by filename.

    Args:
        filename: File name inside this package, e.g. ``"0001_initial.py"``.

    Returns:
        The imported module.

    Raises:
        ImportError: If the file cannot be loaded.
    """
    path = MIGRATIONS_DIR / filename
    module_name = f"{__name__}._{path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load migration {filename!r} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


initial = load_migration("0001_initial.py")
"""The ``0001_initial`` migration module."""

market_recording = load_migration("0002_market_recording.py")
"""The ``0002_market_recording`` migration module."""

__all__ = ["MIGRATIONS_DIR", "initial", "load_migration", "market_recording"]
