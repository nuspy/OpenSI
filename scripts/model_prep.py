"""CLI entrypoint for model-prep tool."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_prep.cli import main

if __name__ == "__main__":
    sys.exit(main())
