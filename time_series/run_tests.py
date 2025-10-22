#!/usr/bin/env python3
"""
Test runner for the demand forecasting project
"""

import subprocess
import sys
import argparse


def run_tests(unit=True, integration=True, coverage=True, parallel=False):
    """Run the test suite with specified options"""

    cmd = ["pytest"]

    if unit and not integration:
        cmd.extend(["-m", "unit"])
    elif integration and not unit:
        cmd.extend(["-m", "integration"])

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term", "--cov-report=html"])

    if parallel:
        cmd.extend(["-n", "auto"])

    cmd.extend(["-v", "tests/"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for demand forecasting project")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    unit = not args.integration_only
    integration = not args.unit_only
    coverage = not args.no_coverage

    exit_code = run_tests(unit, integration, coverage, args.parallel)
    sys.exit(exit_code)