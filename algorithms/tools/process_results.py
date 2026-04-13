#!/usr/bin/env python3
"""
Compatibility wrapper for the consolidated process-results pipeline.

The maintained implementation lives at algorithms/process_results.py. This wrapper
keeps older commands working while avoiding a second copy of the processing logic.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.process_results import main


if __name__ == "__main__":
    main()
