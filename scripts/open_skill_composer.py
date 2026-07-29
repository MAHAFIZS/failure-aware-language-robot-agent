#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COLORS = ["red", "green", "blue"]

WORKSPACE = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.42,
    "y_max": 0.42,
    "z": 0.02575,
}

MAX_STEPS = 6


def move_step(
    object_name: str,
    target: list[float],
) -> dict:
    return {
        "tool": "move_object",
        "object": object_name,
        "target_position": target,
    }


def extract_colors(text: str) -> list[str]:
    return re.findall(
        r"\b(red|green|blue)\b",
        text.lower(),
    )


def compose_plan(instruction: str) -> dict:
    text = instruction.lower().strip()
    steps: list[dict] = []
    goal_type = "unsupported"

    # -------------------------------------------------------------
    # Opposite sides of a reference object
    # -------------------------------------------------------------

    if "opposite sides" in text:
        colors = extract_colors(text)

        if len(colors) >= 3:
            first = colors[0]
            second = colors[1]
            reference = colors[2]
        else:
            first = "red"
            second = "green"
            reference = "blue"

        goal_type = "opposite_sides"

        # Keep blue/reference in the middle.
        center = [0.40, 0.00, WORKSPACE["z"]]

        steps = [
            move_step(reference, center),
            move_step(
                first,
                [0.40, 0.15, WORKSPACE["z"]],
            ),
            move_step(
                second,
                [0.40, -0.15, WORKSPACE["z"]],
            ),
        ]

    # -------------------------------------------------------------
    # Horizontal line
    # -------------------------------------------------------------

    elif (
        "horizontal line" in text
        or "straight line" in text
        or "in a row" in text
    ):
        goal_type = "horizontal_line"

        steps = [
            move_step(
                "red",
                [0.40, 0.16, WORKSPACE["z"]],
            ),
            move_step(
                "green",
                [0.40, 0.00, WORKSPACE["z"]],
            ),
            move_step(
                "blue",
                [0.40, -0.16, WORKSPACE["z"]],
            ),
        ]

    # -------------------------------------------------------------
    # Triangle
    # -------------------------------------------------------------

    elif "triangle" in text:
        goal_type = "triangle"

        steps = [
            move_step(
                "red",
                [0.33, 0.14, WORKSPACE["z"]],
            ),
            move_step(
                "green",
                [0.33, -0.14, WORKSPACE["z"]],
            ),
            move_step(
                "blue",
                [0.50, 0.00, WORKSPACE["z"]],
            ),
        ]

    # -------------------------------------------------------------
    # Explicit relative placement
    # -------------------------------------------------------------

    elif any(
        phrase in text
        for phrase in [
            "beside",
            "next to",
            "left of",
            "right of",
            "in front of",
            "behind",
        ]
    ):
        colors = extract_colors(text)

        if len(colors) >= 2:
            moving = colors[0]
            reference = colors[1]

            if "left of" in text:
                relation = "left_of"
            elif "right of" in text:
                relation = "right_of"
            elif "in front of" in text:
                relation = "in_front_of"
            elif "behind" in text:
                relation = "behind"
            else:
                relation = "beside"

            goal_type = "relative_placement"

            steps = [
                {
                    "tool": "place_relative",
                    "object": moving,
                    "reference_object": reference,
                    "relation": relation,
                }
            ]

    errors: list[str] = []

    if not steps:
        errors.append(
            "The instruction requires a primitive or "
            "arrangement that is not supported yet."
        )

    if len(steps) > MAX_STEPS:
        errors.append(
            f"Plan exceeds the maximum of {MAX_STEPS} steps."
        )

    for index, step in enumerate(steps, start=1):
        object_name = step.get("object")

        if object_name not in COLORS:
            errors.append(
                f"Step {index}: unknown object {object_name}."
            )

        target = step.get("target_position")

        if target is not None:
            x, y, _ = target

            if not (
                WORKSPACE["x_min"]
                <= x
                <= WORKSPACE["x_max"]
            ):
                errors.append(
                    f"Step {index}: target X is outside workspace."
                )

            if not (
                WORKSPACE["y_min"]
                <= y
                <= WORKSPACE["y_max"]
            ):
                errors.append(
                    f"Step {index}: target Y is outside workspace."
                )

    return {
        "schema_version":
            "isaac_open_skill_plan_v0.1",
        "instruction":
            instruction,
        "planner_type":
            "compositional_primitive_skill_planner",
        "goal_type":
            goal_type,
        "approved":
            not errors,
        "step_count":
            len(steps),
        "steps":
            steps,
        "validation_errors":
            errors,
        "capabilities": {
            "persistent_scene_required":
                len(steps) > 1,
            "new_whole_task_script_required":
                False,
            "primitive_tools": [
                "move_object",
                "place_relative",
                "verify_position",
            ],
        },
        "safety_policy": {
            "maximum_steps":
                MAX_STEPS,
            "workspace":
                WORKSPACE,
            "direct_joint_commands_allowed":
                False,
            "unknown_tools_rejected":
                True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/latest_open_skill_plan.json"
        ),
    )

    args = parser.parse_args()

    plan = compose_plan(args.instruction)

    output_path = Path(args.output).expanduser()

    if not output_path.is_absolute():
        output_path = (
            Path(__file__).resolve().parents[1]
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(plan, indent=2))
    print(f"\nPlan saved: {output_path}")

    raise SystemExit(
        0 if plan["approved"] else 2
    )


if __name__ == "__main__":
    main()
