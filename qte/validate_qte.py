"""Run the QTE test suite as a deployment/debug gate."""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path

from . import test_qte


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_qte)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "generated": datetime.now().astimezone().isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
    }
    output = Path(__file__).resolve().parent / "validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
