"""Ensure the project root is importable in tests (so `import config` / `from src...` work)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
