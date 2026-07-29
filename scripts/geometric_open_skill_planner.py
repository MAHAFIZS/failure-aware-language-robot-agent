#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OBJECTS = ["red", "green", "blue"]

WORKSPACE = {
    "x_min": 0.20,
    "x_max": 0.60,
    "y_min": -0.42,
    "y_max": 0.42,
    "z": 0.02575,
}

MAX_STEPS = 6


def extract_requested_order(text: str) -> list[str]:
    text = text.lower()

    colors = re.findall(
        r"\b(red|green|blue)\b",
        text,
    )

    unique_colors: list[str] = []

    for color in colors:
        if color not in unique_colors:
            unique_colors.append(color)

    if len(unique_colors) == 3:
        return unique_colors

    if "rgb" in text:
        return ["red", "green", "blue"]

    if "bgr" in text:
        return ["blue", "green", "red"]

    return OBJECTS.copy()


def move_step(
    object_name: str,
    target_position: list[float],
) -> dict:
    return {
        "tool": "move_object",
        "object": object_name,
        "target_position": target_position,
    }


def horizontal_line(
    order: list[str],
) -> list[dict]:
    targets = [
        [0.40, 0.16, WORKSPACE["z"]],
        [0.40, 0.00, WORKSPACE["z"]],
        [0.40, -0.16, WORKSPACE["z"]],
    ]

    return [
        move_step(object_name, target)
        for object_name, target in zip(order, targets)
    ]


def vertical_line(
    order: list[str],
) -> list[dict]:
    targets = [
        [0.28, 0.00, WORKSPACE["z"]],
        [0.40, 0.00, WORKSPACE["z"]],
        [0.52, 0.00, WORKSPACE["z"]],
    ]

    return [
        move_step(object_name, target)
        for object_name, target in zip(order, targets)
    ]


def diagonal_line(
    order: list[str],
) -> list[dict]:
    targets = [
        [0.28, 0.16, WORKSPACE["z"]],
        [0.40, 0.00, WORKSPACE["z"]],
        [0.52, -0.16, WORKSPACE["z"]],
    ]

    return [
        move_step(object_name, target)
        for object_name, target in zip(order, targets)
    ]


def l_shape(
    order: list[str],
) -> list[dict]:
    targets = [
        [0.32, 0.14, WORKSPACE["z"]],
        [0.32, -0.02, WORKSPACE["z"]],
        [0.48, -0.02, WORKSPACE["z"]],
    ]

    return [
        move_step(object_name, target)
        for object_name, target in zip(order, targets)
    ]


def triangle(
    order: list[str],
) -> list[dict]:
    targets = [
        [0.32, 0.14, WORKSPACE["z"]],
        [0.32, -0.14, WORKSPACE["z"]],
        [0.50, 0.00, WORKSPACE["z"]],
    ]

    return [
        move_step(object_name, target)
        for object_name, target in zip(order, targets)
    ]


def validate_plan(steps: list[dict]) -> list[str]:
    errors: list[str] = []

    if not steps:
        errors.append(
            "No supported geometric arrangement was detected."
        )

    if len(steps) > MAX_STEPS:
        errors.append(
            f"Plan exceeds maximum of {MAX_STEPS} steps."
        )

    seen_objects: set[str] = set()

    for index, step in enumerate(steps, start=1):
        tool = step.get("tool")
        object_name = step.get("object")
        target = step.get("target_position")

        if tool != "move_object":
            errors.append(
                f"Step {index}: unsupported tool {tool}."
            )

        if object_name not in OBJECTS:
            errors.append(
                f"Step {index}: unknown object {object_name}."
            )

        if object_name in seen_objects:
            errors.append(
                f"Step {index}: object {object_name} "
                "appears more than once."
            )

        seen_objects.add(object_name)

        if (
            not isinstance(target, list)
            or len(target) != 3
        ):
            errors.append(
                f"Step {index}: invalid target position."
            )
            continue

        x, y, z = target

        if not (
            WORKSPACE["x_min"]
            <= float(x)
            <= WORKSPACE["x_max"]
        ):
            errors.append(
                f"Step {index}: X target outside workspace."
            )

        if not (
            WORKSPACE["y_min"]
            <= float(y)
            <= WORKSPACE["y_max"]
        ):
            errors.append(
                f"Step {index}: Y target outside workspace."
            )

        if abs(float(z) - WORKSPACE["z"]) > 0.01:
            errors.append(
                f"Step {index}: unsafe placement height."
            )

    # Ensure generated targets do not overlap.
    for first_index in range(len(steps)):
        for second_index in range(
            first_index + 1,
            len(steps),
        ):
            first = steps[first_index][
                "target_position"
            ]
            second = steps[second_index][
                "target_position"
            ]

            dx = float(first[0]) - float(second[0])
            dy = float(first[1]) - float(second[1])

            distance_squared = dx * dx + dy * dy

            if distance_squared < 0.01:
                errors.append(
                    "Two target positions are too close."
                )

    return errors


def compose_geometric_plan(
    instruction: str,
) -> dict:
    text = instruction.lower().strip()
    order = extract_requested_order(text)

    goal_type = "unsupported"
    steps: list[dict] = []

    if (
        "l shape" in text
        or "l-shape" in text
        or "letter l" in text
    ):
        goal_type = "l_shape"
        steps = l_shape(order)

    elif (
        "diagonal" in text
        or "slanted line" in text
    ):
        goal_type = "diagonal_line"
        steps = diagonal_line(order)

    elif (
        "vertical line" in text
        or "vertical row" in text
        or "top to bottom" in text
    ):
        goal_type = "vertical_line"
        steps = vertical_line(order)

    elif (
        "horizontal line" in text
        or "horizontal row" in text
        or "left to right" in text
        or "in a row" in text
        or "straight line" in text
    ):
        goal_type = "horizontal_line"
        steps = horizontal_line(order)

    elif "triangle" in text:
        goal_type = "triangle"
        steps = triangle(order)

    errors = validate_plan(steps)

    return {
        "schema_version":
            "isaac_geometric_open_skill_plan_v0.1",
        "instruction":
            instruction,
        "planner_type":
            "deterministic_geometric_planner",
        "goal_type":
            goal_type,
        "object_order":
            order,
        "approved":
            not errors,
        "step_count":
            len(steps),
        "steps":
            steps,
        "validation_errors":
            errors,
        "capabilities": {
            "coordinate_generation":
                True,
            "persistent_scene_required":
                len(steps) > 1,
            "new_task_script_required":
                False,
            "approved_primitive":
                "move_object",
        },
        "safety_policy": {
            "workspace":
                WORKSPACE,
            "maximum_steps":
                MAX_STEPS,
            "minimum_target_spacing_m":
                0.10,
            "direct_joint_commands_allowed":
                False,
            "unknown_objects_rejected":
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
            "results/"
            "latest_geometric_open_skill_plan.json"
        ),
    )

    args = parser.parse_args()

    plan = compose_geometric_plan(
        args.instruction
    )

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
