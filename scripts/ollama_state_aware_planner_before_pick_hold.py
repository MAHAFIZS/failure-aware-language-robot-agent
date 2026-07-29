from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from stack_scene_graph import (
    KNOWN_OBJECTS,
    plan_place_on_object,
    plan_tower,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "agent_status.json"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "latest_chat_ollama_plan.json"
)

GEOMETRIC_PLANNER = (
    PROJECT_ROOT
    / "scripts"
    / "ollama_open_skill_planner.py"
)

OLLAMA_URL = "http://localhost:11434/api/chat"

SAFE_TOWER_XY = [0.50, 0.00]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def extract_positions(
    status: dict[str, Any],
) -> dict[str, list[float]]:
    """
    Read the latest persistent Isaac scene from idle or
    completed status payloads.
    """

    scene = status.get("scene")

    if not scene:
        scene = (
            status
            .get("result", {})
            .get("final_scene")
        )

    if not isinstance(scene, dict):
        raise RuntimeError(
            "No current Isaac scene was found in the status file."
        )

    positions: dict[str, list[float]] = {}

    for object_name in KNOWN_OBJECTS:
        object_data = scene.get(object_name)

        if not isinstance(object_data, dict):
            continue

        position = object_data.get("position")

        if (
            isinstance(position, list)
            and len(position) == 3
        ):
            positions[object_name] = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ]

    if not positions:
        raise RuntimeError(
            "No cube positions were found in the Isaac status."
        )

    return positions


def call_ollama_intent_parser(
    instruction: str,
    model: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "make_tower",
                    "place_on_object",
                    "geometric_arrangement",
                    "pick_object",
                    "unsupported",
                ],
            },
            "moving_object": {
                "type": ["string", "null"],
                "enum": [
                    *KNOWN_OBJECTS,
                    None,
                ],
            },
            "support_object": {
                "type": ["string", "null"],
                "enum": [
                    *KNOWN_OBJECTS,
                    None,
                ],
            },
            "tower_bottom_to_top": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": KNOWN_OBJECTS,
                },
                "maxItems": 8,
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "intent",
            "moving_object",
            "support_object",
            "tower_bottom_to_top",
            "reason",
        ],
    }

    system_prompt = f"""
You are an intent parser for an Isaac Sim robot.

Available cube colors:
{", ".join(KNOWN_OBJECTS)}

Classify the user command into exactly one intent.

Rules:

1. make_tower
Use when the user asks for a tower, stack, or vertical ordering.
The array tower_bottom_to_top must be ordered from the bottom
cube to the top cube.

Example:
"Make a tower with red at the bottom, green in the middle,
and blue on top."

Result:
intent = make_tower
tower_bottom_to_top = ["red", "green", "blue"]

2. place_on_object
Use when one cube must be placed on another cube.

Example:
"Place blue on yellow."

Result:
intent = place_on_object
moving_object = "blue"
support_object = "yellow"

3. geometric_arrangement
Use for letters, numbers, lines, circles, arrows, V shapes,
P shapes, or other tabletop arrangements.

4. pick_object
Use for commands such as "pick red", "grasp blue",
or "hold yellow".

5. unsupported
Use when the command cannot be interpreted safely.

Do not invent colors.
Do not add cubes that the user did not request for a tower.
"""

    payload = {
        "model": model,
        "stream": False,
        "format": schema,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": instruction,
            },
        ],
        "options": {
            "temperature": 0.0,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        ollama_response = json.loads(
            response.read().decode("utf-8")
        )

    content = (
        ollama_response
        .get("message", {})
        .get("content", "")
    )

    if not content:
        raise RuntimeError(
            "Ollama returned no structured intent."
        )

    return json.loads(content)


def rejected_plan(
    instruction: str,
    intent: str,
    error: str,
    parsed_intent: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version":
            "isaac_state_aware_plan_v0.1",
        "instruction":
            instruction,
        "planner_type":
            "ollama_intent_plus_deterministic_planner",
        "intent":
            intent,
        "approved":
            False,
        "step_count":
            0,
        "steps":
            [],
        "validation_errors":
            [error],
        "parsed_intent":
            parsed_intent,
    }


def run_geometric_planner(
    instruction: str,
    model: str,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(GEOMETRIC_PLANNER),
        "--instruction",
        instruction,
        "--model",
        model,
        "--output",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    if not output_path.exists():
        raise RuntimeError(
            result.stderr
            or result.stdout
            or "Geometric planner produced no output."
        )

    return read_json(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instruction",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="qwen3:8b",
    )

    parser.add_argument(
        "--status-file",
        default=str(DEFAULT_STATUS_FILE),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
    )

    args = parser.parse_args()

    instruction = args.instruction.strip()
    status_file = Path(args.status_file).expanduser()
    output_file = Path(args.output).expanduser()

    parsed = call_ollama_intent_parser(
        instruction,
        args.model,
    )

    intent = parsed["intent"]

    if intent == "geometric_arrangement":
        plan = run_geometric_planner(
            instruction,
            args.model,
            output_file,
        )

        print(json.dumps(plan, indent=2))
        return

    if intent == "pick_object":
        plan = rejected_plan(
            instruction,
            intent,
            (
                "Pick-and-hold execution is not connected yet. "
                "Tower and place-on-object commands are supported."
            ),
            parsed,
        )

        write_json_atomic(output_file, plan)
        print(json.dumps(plan, indent=2))
        return

    if intent == "unsupported":
        plan = rejected_plan(
            instruction,
            intent,
            parsed.get(
                "reason",
                "Unsupported command.",
            ),
            parsed,
        )

        write_json_atomic(output_file, plan)
        print(json.dumps(plan, indent=2))
        return

    if not status_file.exists():
        plan = rejected_plan(
            instruction,
            intent,
            "The persistent Isaac status file does not exist.",
            parsed,
        )

        write_json_atomic(output_file, plan)
        print(json.dumps(plan, indent=2))
        return

    status = read_json(status_file)
    positions = extract_positions(status)

    if intent == "place_on_object":
        moving_object = parsed.get("moving_object")
        support_object = parsed.get("support_object")

        if not moving_object or not support_object:
            plan = rejected_plan(
                instruction,
                intent,
                (
                    "Both moving_object and support_object "
                    "are required."
                ),
                parsed,
            )
        else:
            plan = plan_place_on_object(
                positions,
                moving_object,
                support_object,
            )

    elif intent == "make_tower":
        tower_order = parsed.get(
            "tower_bottom_to_top",
            [],
        )

        if len(tower_order) < 2:
            plan = rejected_plan(
                instruction,
                intent,
                (
                    "A tower requires at least two explicitly "
                    "named cubes."
                ),
                parsed,
            )
        else:
            plan = plan_tower(
                positions,
                tower_order,
                SAFE_TOWER_XY,
            )

    else:
        plan = rejected_plan(
            instruction,
            intent,
            "No planner is available for this intent.",
            parsed,
        )

    plan["schema_version"] = (
        "isaac_state_aware_plan_v0.1"
    )
    plan["instruction"] = instruction
    plan["planner_type"] = (
        "ollama_intent_plus_deterministic_planner"
    )
    plan["parsed_intent"] = parsed
    plan["step_count"] = len(
        plan.get("steps", [])
    )

    write_json_atomic(output_file, plan)

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
