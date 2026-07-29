#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COLORS = {"red", "green", "blue"}

SUPPORTED_SKILLS = {
    "pick_place_relative",
    "sort_all",
    "make_tower",
}

SUPPORTED_RELATIONS = {
    "left_of",
    "right_of",
    "beside",
    "in_front_of",
    "behind",
    "on_top_of",
}


def split_instruction(instruction: str) -> list[str]:
    parts = re.split(
        r"\b(?:and then|then|after that)\b",
        instruction,
        flags=re.IGNORECASE,
    )

    return [
        part.strip(" ,.")
        for part in parts
        if part.strip(" ,.")
    ]


def detect_relation(text: str) -> str | None:
    text = text.lower()

    relation_phrases = [
        ("on top of", "on_top_of"),
        ("stack on", "on_top_of"),
        ("to the left of", "left_of"),
        ("left of", "left_of"),
        ("to the right of", "right_of"),
        ("right of", "right_of"),
        ("in front of", "in_front_of"),
        ("behind", "behind"),
        ("next to", "beside"),
        ("beside", "beside"),
        ("near", "beside"),
        ("close to", "beside"),
    ]

    for phrase, relation in relation_phrases:
        if phrase in text:
            return relation

    return None


def extract_colors(text: str) -> list[str]:
    return re.findall(
        r"\b(red|green|blue)\b",
        text.lower(),
    )


def parse_step(text: str) -> dict:
    lower = text.lower()

    if (
        "sort" in lower
        or "matching bins" in lower
        or "matching trays" in lower
    ):
        return {
            "skill": "sort_all",
            "parameters": {},
            "source_text": text,
        }

    if (
        "tower" in lower
        or "stack all" in lower
        or "pile all" in lower
    ):
        return {
            "skill": "make_tower",
            "parameters": {
                "objects": [
                    "red",
                    "green",
                    "blue",
                ]
            },
            "source_text": text,
        }

    colors = extract_colors(text)
    relation = detect_relation(text)

    movement_words = (
        "pick",
        "place",
        "put",
        "move",
        "position",
        "set",
        "bring",
    )

    if (
        any(word in lower for word in movement_words)
        and len(colors) >= 2
        and relation is not None
    ):
        return {
            "skill": "pick_place_relative",
            "parameters": {
                "object": colors[0],
                "reference_object": colors[1],
                "relation": relation,
            },
            "source_text": text,
        }

    return {
        "skill": "unsupported",
        "parameters": {},
        "source_text": text,
        "reason": (
            "No supported robot skill could be derived."
        ),
    }


def validate_step(step: dict) -> list[str]:
    errors: list[str] = []

    skill = step.get("skill")

    if skill not in SUPPORTED_SKILLS:
        errors.append(
            f"Unsupported skill: {skill}"
        )
        return errors

    parameters = step.get("parameters", {})

    if skill == "pick_place_relative":
        moving_object = parameters.get("object")
        reference_object = parameters.get(
            "reference_object"
        )
        relation = parameters.get("relation")

        if moving_object not in COLORS:
            errors.append(
                f"Unknown object: {moving_object}"
            )

        if reference_object not in COLORS:
            errors.append(
                f"Unknown reference object: "
                f"{reference_object}"
            )

        if moving_object == reference_object:
            errors.append(
                "Moving and reference objects "
                "must be different."
            )

        if relation not in SUPPORTED_RELATIONS:
            errors.append(
                f"Unsupported relation: {relation}"
            )

    return errors


def create_plan(instruction: str) -> dict:
    raw_steps = split_instruction(instruction)

    parsed_steps = [
        parse_step(step)
        for step in raw_steps
    ]

    validation_errors: list[str] = []

    for index, step in enumerate(
        parsed_steps,
        start=1,
    ):
        for error in validate_step(step):
            validation_errors.append(
                f"Step {index}: {error}"
            )

    approved = (
        bool(parsed_steps)
        and not validation_errors
    )

    return {
        "schema_version":
            "isaac_open_language_plan_v0.1",
        "instruction":
            instruction,
        "planner_type":
            "open_language_closed_skill_planner",
        "approved":
            approved,
        "step_count":
            len(parsed_steps),
        "steps":
            parsed_steps,
        "validation_errors":
            validation_errors,
        "safety_policy": {
            "maximum_steps": 6,
            "supported_objects":
                sorted(COLORS),
            "supported_skills":
                sorted(SUPPORTED_SKILLS),
            "unknown_skills_rejected":
                True,
            "direct_joint_commands_allowed":
                False,
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
            "latest_open_language_plan.json"
        ),
    )

    args = parser.parse_args()

    plan = create_plan(args.instruction)

    if plan["step_count"] > 6:
        plan["approved"] = False
        plan["validation_errors"].append(
            "Plan exceeds maximum of 6 steps."
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
