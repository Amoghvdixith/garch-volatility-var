"""Run the test suite and save JUnit XML into a results folder.

Usage (from the project root):

    python scripts/run_tests.py --results-dir results/synthetic_baseline

Equivalent to ``python -m pytest -q --junitxml=<dir>/test_results.xml``,
followed by removal of the machine ``hostname`` attribute that pytest writes
into the XML, so the saved file contains no machine-identifying detail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run pytest and save sanitised JUnit XML.")
    p.add_argument("--results-dir", required=True, help="Folder for test_results.xml, e.g. results/synthetic_baseline")
    args = p.parse_args(argv)
    out = PROJECT_ROOT / args.results_dir / "test_results.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={out}", "-o", "junit_family=xunit2"],
        cwd=PROJECT_ROOT,
    )
    if out.is_file():
        tree = ET.parse(out)
        for el in tree.getroot().iter():
            el.attrib.pop("hostname", None)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        print(f"Saved {Path(args.results_dir).as_posix()}/test_results.xml (exit code {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
