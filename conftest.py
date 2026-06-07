"""Pytest bootstrap: put `src/` on sys.path so `import htr_sp1...` works without
installing the package. Keeps local test runs friction-free for the thesis team.
"""
import sys
from pathlib import Path

# src/ sits next to this file; prepend it so the htr_sp1 package is importable.
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
