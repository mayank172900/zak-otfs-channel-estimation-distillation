from __future__ import annotations

import sys
from pathlib import Path


PHYSICS_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PHYSICS_SRC) not in sys.path:
    sys.path.insert(0, str(PHYSICS_SRC))
