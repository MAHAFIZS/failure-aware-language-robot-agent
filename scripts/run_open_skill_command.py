#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from open_skill_composer import compose_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path.home() / "isaac-sim" / "python.sh"

PLAN_PATH = (
    PROJECT_ROOT
    / "results"
    / "latest_open_skill_plan.json"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "latest_open_skill_execution.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    plan = compose_plan(args.instruction)

    PLAN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLAN_PATH.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    print("\n[Open Skill Command] Generated plan:")
    print(json.dumps(plan, indent=2))

    if not plan.get("approved"):
        print(
            "\n[Open Skill Command] "
            "Plan rejected by validator."
        )
        raise SystemExit(2)

    command = [
        str(ISAAC_PYTHON),
        str(
            PROJECT_ROOT
            / "apps"
            / "franka_open_skill_agent.py"
        ),
        "--plan-json",
        str(PLAN_PATH),
        "--seed",
        str(args.seed),
        "--result-json",
        str(RESULT_PATH),
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
