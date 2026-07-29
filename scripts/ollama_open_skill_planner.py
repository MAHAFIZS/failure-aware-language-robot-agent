#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

OBJECTS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "white",
]
WORKSPACE = {
    "x_min": 0.32,
    "x_max": 0.65,
    "y_min": -0.38,
    "y_max": 0.38,
    "z": 0.02575,
}
MIN_SPACING = 0.10


SCHEMA = {
    "type": "object",
    "properties": {
        "goal_description": {
            "type": "string"
        },
        "approximation_note": {
            "type": ["string", "null"]
        },
        "steps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["move_object"]
                    },
                    "object": {
                        "type": "string",
                        "enum": OBJECTS
                    },
                    "target_position": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 8,
                        "items": {
                            "type": "number"
                        }
                    }
                },
                "required": [
                    "tool",
                    "object",
                    "target_position"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "goal_description",
        "approximation_note",
        "steps"
    ],
    "additionalProperties": False
}


SYSTEM_PROMPT = """
You are a geometric planner for a Franka robot in Isaac Sim.

Important command rules:
- Move only the objects explicitly requested by the user.
- Never move every cube unless the user explicitly requests all cubes.
- A command such as "pick blue box" has no destination and cannot be executed with move_object.
- For pick-only, hold, lift, grasp, or release commands, return approved false with zero steps because those tools are not currently available.
- For a single-object move command, generate exactly one step.
- Do not invent an arrangement when the user only names one object.

Available objects:
- red cube
- green cube
- blue cube
- yellow cube
- orange cube
- purple cube
- cyan cube
- white cube

Allowed primitive:
- move_object

Workspace:
- X between 0.32 and 0.65 metres
- Y between -0.38 and 0.38 metres
- Z exactly 0.02575 metres

Generate at most eight steps.
Use each cube at most once.
Keep all target positions at least 0.10 metres apart.
Never output joint commands, forces, velocities, code, or unsupported tools.
For letters, numbers and shapes, use as many available cubes as needed, up to eight. Generate a recognizable discrete approximation and explain any limitation.
Output only valid JSON matching the provided schema.
""".strip()


def call_ollama(
    instruction: str,
    model: str,
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {
            "temperature": 0
        },
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": instruction
            }
        ]
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        raw = json.loads(
            response.read().decode("utf-8")
        )

    content = raw["message"]["content"]
    return json.loads(content)


def repair_generated_plan(plan: dict) -> dict:
    """
    Repair common local-LLM formatting mistakes without allowing
    unsupported tools or unsafe workspace coordinates.
    """
    repaired = dict(plan)
    original_steps = plan.get("steps", [])

    repaired_steps = []
    available_objects = OBJECTS.copy()

    for index, original_step in enumerate(original_steps):
        if index >= len(available_objects):
            break

        step = dict(original_step)

        # Assign every movement to a unique available cube.
        step["tool"] = "move_object"
        step["object"] = available_objects[index]

        target = step.get("target_position", [])

        if not isinstance(target, list) or len(target) != 3:
            continue

        try:
            x = float(target[0])
            y = float(target[1])
        except (TypeError, ValueError):
            continue

        # Clamp the generated XY point to the validated workspace.
        x = min(
            max(x, WORKSPACE["x_min"]),
            WORKSPACE["x_max"],
        )
        y = min(
            max(y, WORKSPACE["y_min"]),
            WORKSPACE["y_max"],
        )

        # Isaac tabletop height is deterministic.
        z = WORKSPACE["z"]

        step["target_position"] = [x, y, z]
        repaired_steps.append(step)

    # Repair duplicate or too-close target positions while
    # preserving the LLM geometry as closely as possible.
    safe_steps = []
    occupied_positions = []

    candidate_offsets = [
        (0.0, 0.0),
        (0.0, 0.11),
        (0.0, -0.11),
        (0.11, 0.0),
        (-0.11, 0.0),
        (0.11, 0.11),
        (0.11, -0.11),
        (-0.11, 0.11),
        (-0.11, -0.11),
        (0.0, 0.22),
        (0.0, -0.22),
        (0.22, 0.0),
        (-0.22, 0.0),
    ]

    for step in repaired_steps:
        original_x = float(
            step["target_position"][0]
        )
        original_y = float(
            step["target_position"][1]
        )

        selected_position = None

        for offset_x, offset_y in candidate_offsets:
            candidate_x = min(
                max(
                    original_x + offset_x,
                    WORKSPACE["x_min"],
                ),
                WORKSPACE["x_max"],
            )

            candidate_y = min(
                max(
                    original_y + offset_y,
                    WORKSPACE["y_min"],
                ),
                WORKSPACE["y_max"],
            )

            candidate_is_safe = True

            for occupied_x, occupied_y in occupied_positions:
                distance = (
                    (candidate_x - occupied_x) ** 2
                    + (candidate_y - occupied_y) ** 2
                ) ** 0.5

                if distance + 1e-6 < MIN_SPACING:
                    candidate_is_safe = False
                    break

            if candidate_is_safe:
                selected_position = (
                    candidate_x,
                    candidate_y,
                )
                break

        if selected_position is None:
            continue

        selected_x, selected_y = selected_position

        step["target_position"] = [
            selected_x,
            selected_y,
            WORKSPACE["z"],
        ]

        occupied_positions.append(
            (selected_x, selected_y)
        )

        safe_steps.append(step)

    repaired["steps"] = safe_steps

    note = repaired.get("approximation_note")

    repair_note = (
        "Planner output was normalized by the deterministic "
        "safety layer: unique cube assignments and validated "
        "table height were applied."
    )

    repaired["approximation_note"] = (
        f"{note} {repair_note}"
        if note
        else repair_note
    )

    return repaired



def validate(plan: dict) -> list[str]:
    errors = []
    steps = plan.get("steps", [])
    seen = set()
    positions = []

    if not steps:
        errors.append("No movement steps generated.")

    for index, step in enumerate(steps, start=1):
        object_name = step.get("object")
        target = step.get("target_position")

        if step.get("tool") != "move_object":
            errors.append(
                f"Step {index}: unsupported tool."
            )

        if object_name not in OBJECTS:
            errors.append(
                f"Step {index}: unknown object."
            )

        if object_name in seen:
            errors.append(
                f"Step {index}: duplicate object."
            )

        seen.add(object_name)

        if not isinstance(target, list) or len(target) != 3:
            errors.append(
                f"Step {index}: invalid target."
            )
            continue

        x, y, z = map(float, target)

        if not WORKSPACE["x_min"] <= x <= WORKSPACE["x_max"]:
            errors.append(
                f"Step {index}: X outside workspace."
            )

        if not WORKSPACE["y_min"] <= y <= WORKSPACE["y_max"]:
            errors.append(
                f"Step {index}: Y outside workspace."
            )

        if abs(z - WORKSPACE["z"]) > 0.001:
            errors.append(
                f"Step {index}: unsafe Z value."
            )

        positions.append((x, y))

    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            x1, y1 = positions[first]
            x2, y2 = positions[second]

            distance = (
                (x1 - x2) ** 2
                + (y1 - y2) ** 2
            ) ** 0.5

            if distance + 1e-6 < MIN_SPACING:
                errors.append(
                    f"Targets too close: {distance:.3f} m."
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        required=True
    )

    parser.add_argument(
        "--model",
        default="qwen3:8b"
    )

    parser.add_argument(
        "--output",
        default=(
            "results/"
            "latest_ollama_open_skill_plan.json"
        )
    )

    args = parser.parse_args()

    generated_raw = call_ollama(
        args.instruction,
        args.model,
    )

    generated = repair_generated_plan(
        generated_raw
    )

    errors = validate(generated)

    result = {
        "schema_version":
            "isaac_ollama_open_skill_plan_v0.1",
        "instruction":
            args.instruction,
        "planner_type":
            "local_llm_geometric_planner",
        "model":
            args.model,
        "goal_type":
            "llm_generated_geometry",
        "goal_description":
            generated.get("goal_description"),
        "approximation_note":
            generated.get("approximation_note"),
        "approved":
            not errors,
        "step_count":
            len(generated.get("steps", [])),
        "steps":
            generated.get("steps", []),
        "validation_errors":
            errors,
        "safety_policy": {
            "approved_tools": ["move_object"],
            "known_objects": OBJECTS,
            "workspace": WORKSPACE,
            "minimum_target_spacing_m":
                MIN_SPACING,
            "direct_joint_commands_allowed":
                False
        }
    }

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))
    print(f"\nPlan saved: {output_path}")


if __name__ == "__main__":
    main()
