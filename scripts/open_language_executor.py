#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from open_language_planner import create_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path.home() / "isaac-sim" / "python.sh"

RESULT_DIR = PROJECT_ROOT / "results"


def relative_instruction(parameters: dict) -> str:
    relation_text = {
        "left_of": "left of",
        "right_of": "right of",
        "beside": "beside",
        "in_front_of": "in front of",
        "behind": "behind",
        "on_top_of": "on top of",
    }

    moving = parameters["object"]
    reference = parameters["reference_object"]
    relation = relation_text[parameters["relation"]]

    return (
        f"Pick the {moving} box and place it "
        f"{relation} the {reference} box."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan and execute an approved Isaac Sim skill."
    )

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

    plan = create_plan(args.instruction)

    print("\n[Open Agent] Generated plan:")
    print(json.dumps(plan, indent=2))

    if not plan["approved"]:
        print("\n[Open Agent] Plan rejected by safety validator.")
        raise SystemExit(2)

    if plan["step_count"] != 1:
        print(
            "\n[Open Agent] Multi-step execution is not enabled yet. "
            "The plan was accepted, but execution was stopped to avoid "
            "resetting the simulator state between scripts."
        )
        raise SystemExit(3)

    step = plan["steps"][0]
    skill = step["skill"]
    parameters = step["parameters"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if skill == "pick_place_relative":
        instruction = relative_instruction(parameters)

        command = [
            str(ISAAC_PYTHON),
            str(PROJECT_ROOT / "apps" / "franka_agent_pick_place.py"),
            "--instruction",
            instruction,
            "--seed",
            str(args.seed),
            "--result-json",
            str(RESULT_DIR / "latest_isaac_agent_result.json"),
        ]

    elif skill == "sort_all":
        command = [
            str(ISAAC_PYTHON),
            str(PROJECT_ROOT / "apps" / "franka_multi_box_sort.py"),
            "--instruction",
            args.instruction,
            "--seed",
            str(args.seed),
            "--result-json",
            str(RESULT_DIR / "latest_isaac_sort_result.json"),
        ]

    elif skill == "make_tower":
        command = [
            str(ISAAC_PYTHON),
            str(PROJECT_ROOT / "apps" / "franka_make_tower.py"),
            "--instruction",
            args.instruction,
            "--seed",
            str(args.seed),
            "--result-json",
            str(RESULT_DIR / "latest_isaac_tower_result.json"),
        ]

    else:
        print(f"[Open Agent] Unsupported executor skill: {skill}")
        raise SystemExit(4)

    print(f"\n[Open Agent] Executing skill: {skill}")
    print("[Open Agent] Command:")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
