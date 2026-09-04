#!/usr/bin/env python3
"""
Phase 12: Refresh the cached pytest summary the dashboard displays.

Runs the REAL test suite once and caches count/pass-fail to
`artifacts/dashboard_test_summary.json`. The dashboard itself never runs
pytest live (far too slow for an interactive page load) — it only reads
this cache, clearly labeled with the timestamp of the run that produced
it, so it's honestly "as of last recorded run," not a live count.

Usage:
    python scripts/refresh_dashboard_test_summary.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "artifacts" / "dashboard_test_summary.json"


def main() -> int:
    print("Running the real test suite (this may take a little while)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    print(output[-2000:])

    match = re.search(r"(\d+) passed(?:, (\d+) failed)?", output)
    n_passed = int(match.group(1)) if match else None
    fail_match = re.search(r"(\d+) failed", output)
    n_failed = int(fail_match.group(1)) if fail_match else 0

    summary = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "all_passed": (result.returncode == 0),
        "note": "Cached result of a real `pytest -q` run — the dashboard reads this file, it does not run pytest live.",
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {CACHE_PATH}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
