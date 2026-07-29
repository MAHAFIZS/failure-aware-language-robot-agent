from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWN_OBJECTS = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "white",
]

TABLE_Z = 0.02575

# Safe manipulation region, moved away from the robot base.
ROBOT_X_MIN = 0.34
ROBOT_X_MAX = 0.64
ROBOT_Y_MIN = -0.32
ROBOT_Y_MAX = 0.32

IMAGE_SIZE = 320
FONT_SIZE = 260

MIN_TARGET_SPACING_M = 0.085

FONT_CANDIDATES = [
    Path(
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf"
    ),
    Path(
        "/usr/share/fonts/truetype/liberation2/"
        "LiberationSans-Bold.ttf"
    ),
    Path(
        "/usr/share/fonts/truetype/freefont/"
        "FreeSansBold.ttf"
    ),
]


def find_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No supported bold system font was found."
    )


def extract_requested_letter(
    instruction: str,
) -> str | None:
    text = instruction.upper().strip()

    # Prefer explicit LETTER/CAPITAL forms, including quotes.
    patterns = [
        r'\bCAPITAL\s+LETTER\s+["\\\']?([A-Z])["\\\']?\b',
        r'\bCAPITAL\s+["\\\']?([A-Z])["\\\']?\b',
        r'\bLETTER\s+["\\\']?([A-Z])["\\\']?\b',
        r'\bMAKE\s+(?:A\s+)?(?:CAPITAL\s+)?(?:LETTER\s+)?["\\\']?([A-Z])["\\\']?\b',
        r'\bCREATE\s+(?:A\s+)?(?:CAPITAL\s+)?(?:LETTER\s+)?["\\\']?([A-Z])["\\\']?\b',
        r'\bFORM\s+(?:A\s+)?(?:CAPITAL\s+)?(?:LETTER\s+)?["\\\']?([A-Z])["\\\']?\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return None


def render_letter_mask(
    letter: str,
) -> np.ndarray:
    font_path = find_font()

    font = ImageFont.truetype(
        str(font_path),
        FONT_SIZE,
    )

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        0,
    )

    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox(
        (0, 0),
        letter,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (
        IMAGE_SIZE - text_width
    ) / 2.0 - bbox[0]

    y = (
        IMAGE_SIZE - text_height
    ) / 2.0 - bbox[1]

    draw.text(
        (x, y),
        letter,
        fill=255,
        font=font,
    )

    mask = np.asarray(image) > 80

    if not np.any(mask):
        raise RuntimeError(
            f"Rendering letter {letter} produced an empty mask."
        )

    return mask


def extract_boundary_pixels(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Select mask pixels that have at least one non-mask neighbour.
    This approximates the visible letter outline without requiring
    an external computer-vision package.
    """

    padded = np.pad(
        mask,
        1,
        mode="constant",
        constant_values=False,
    )

    interior = np.ones_like(mask, dtype=bool)

    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue

            neighbour = padded[
                row_offset:
                    row_offset + mask.shape[0],
                column_offset:
                    column_offset + mask.shape[1],
            ]

            interior &= neighbour

    boundary = mask & ~interior

    rows, columns = np.nonzero(boundary)

    return np.column_stack(
        [columns, rows]
    ).astype(float)


def normalize_points(
    points: np.ndarray,
) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    span = np.maximum(
        maximum - minimum,
        1.0,
    )

    return (
        points - minimum
    ) / span


def farthest_point_sampling(
    points: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Select spatially distributed points from the rendered outline.
    """

    if len(points) < count:
        raise RuntimeError(
            "Not enough rendered pixels to sample the letter."
        )

    centre = np.mean(points, axis=0)

    first_index = int(
        np.argmax(
            np.linalg.norm(
                points - centre,
                axis=1,
            )
        )
    )

    selected_indices = [first_index]

    minimum_distances = np.linalg.norm(
        points - points[first_index],
        axis=1,
    )

    while len(selected_indices) < count:
        next_index = int(
            np.argmax(minimum_distances)
        )

        selected_indices.append(next_index)

        distances = np.linalg.norm(
            points - points[next_index],
            axis=1,
        )

        minimum_distances = np.minimum(
            minimum_distances,
            distances,
        )

    return points[selected_indices]


def robot_distance(
    first: list[float],
    second: list[float],
) -> float:
    return math.hypot(
        first[0] - second[0],
        first[1] - second[1],
    )


def map_to_robot_workspace(
    normalized_points: np.ndarray,
) -> list[list[float]]:
    targets: list[list[float]] = []

    for normalized_x, normalized_y in normalized_points:
        robot_x = (
            ROBOT_X_MIN
            + normalized_x
            * (ROBOT_X_MAX - ROBOT_X_MIN)
        )

        # Image Y grows downward, robot layout Y grows upward.
        robot_y = (
            ROBOT_Y_MAX
            - normalized_y
            * (ROBOT_Y_MAX - ROBOT_Y_MIN)
        )

        targets.append(
            [
                round(float(robot_x), 5),
                round(float(robot_y), 5),
                TABLE_Z,
            ]
        )

    return targets


def improve_target_spacing(
    candidate_points: np.ndarray,
    all_points: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Fallback spacing optimisation. It repeatedly selects the point
    that is farthest from all already-selected targets.
    """

    selected = [
        candidate_points[0]
    ]

    while len(selected) < count:
        selected_array = np.asarray(selected)

        distances = np.linalg.norm(
            all_points[:, None, :]
            - selected_array[None, :, :],
            axis=2,
        )

        minimum_distances = distances.min(axis=1)

        next_index = int(
            np.argmax(minimum_distances)
        )

        selected.append(
            all_points[next_index]
        )

    return np.asarray(selected)


def build_procedural_letter_plan(
    instruction: str,
    letter: str,
) -> dict[str, Any]:
    mask = render_letter_mask(letter)

    boundary_points = extract_boundary_pixels(mask)

    normalized_boundary = normalize_points(
        boundary_points
    )

    sampled_points = farthest_point_sampling(
        normalized_boundary,
        len(KNOWN_OBJECTS),
    )

    targets = map_to_robot_workspace(
        sampled_points
    )

    minimum_spacing = min(
        robot_distance(
            targets[first],
            targets[second],
        )
        for first in range(len(targets))
        for second in range(
            first + 1,
            len(targets),
        )
    )

    if minimum_spacing < MIN_TARGET_SPACING_M:
        sampled_points = improve_target_spacing(
            sampled_points,
            normalized_boundary,
            len(KNOWN_OBJECTS),
        )

        targets = map_to_robot_workspace(
            sampled_points
        )

        minimum_spacing = min(
            robot_distance(
                targets[first],
                targets[second],
            )
            for first in range(len(targets))
            for second in range(
                first + 1,
                len(targets),
            )
        )

    if minimum_spacing < 0.075:
        return {
            "schema_version":
                "isaac_procedural_letter_plan_v0.1",
            "instruction":
                instruction,
            "planner_type":
                "procedural_font_to_points",
            "letter":
                letter,
            "approved":
                False,
            "step_count":
                0,
            "steps":
                [],
            "validation_errors": [
                (
                    "The generated letter points could not be "
                    "separated safely inside the workspace."
                )
            ],
        }

    steps = []

    for object_name, target in zip(
        KNOWN_OBJECTS,
        targets,
    ):
        steps.append(
            {
                "tool": "move_object",
                "object": object_name,
                "target_position": target,
            }
        )

    return {
        "schema_version":
            "isaac_procedural_letter_plan_v0.1",
        "instruction":
            instruction,
        "planner_type":
            "procedural_font_to_points",
        "goal_type":
            "procedural_letter",
        "letter":
            letter,
        "goal_description":
            (
                f"Approximate capital {letter} using eight "
                "points sampled from a rendered font outline."
            ),
        "approximation_note":
            (
                "The letter was generated procedurally from a "
                "system font. No letter-specific robot coordinates "
                "or A-Z templates were used."
            ),
        "approved":
            True,
        "step_count":
            len(steps),
        "steps":
            steps,
        "minimum_target_spacing_m":
            minimum_spacing,
        "validation_errors":
            [],
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
            PROJECT_ROOT
            / "results"
            / "latest_procedural_letter_plan.json"
        ),
    )

    args = parser.parse_args()

    letter = extract_requested_letter(
        args.instruction
    )

    if letter is None:
        plan = {
            "schema_version":
                "isaac_procedural_letter_plan_v0.1",
            "instruction":
                args.instruction,
            "planner_type":
                "procedural_font_to_points",
            "approved":
                False,
            "step_count":
                0,
            "steps":
                [],
            "validation_errors": [
                "No single capital letter was identified."
            ],
        }
    else:
        plan = build_procedural_letter_plan(
            args.instruction,
            letter,
        )

    output_path = Path(args.output).expanduser()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()


def extract_requested_digit(
    instruction: str,
) -> str | None:
    text = instruction.upper().strip()

    word_to_digit = {
        "ZERO": "0",
        "ONE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
    }

    patterns = [
        r"\bNUMBER\s+([0-9])\b",
        r"\bDIGIT\s+([0-9])\b",
        r"\bMAKE\s+(?:THE\s+)?(?:NUMBER\s+|DIGIT\s+)?([0-9])\b",
        r"\bCREATE\s+(?:THE\s+)?(?:NUMBER\s+|DIGIT\s+)?([0-9])\b",
        r"\bFORM\s+(?:THE\s+)?(?:NUMBER\s+|DIGIT\s+)?([0-9])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    for word, digit in word_to_digit.items():
        if re.search(
            rf"\b(?:NUMBER|DIGIT)\s+{word}\b",
            text,
        ):
            return digit

    return None


def build_procedural_digit_plan(
    instruction: str,
    digit: str,
) -> dict[str, Any]:
    if digit not in "0123456789" or len(digit) != 1:
        return {
            "schema_version":
                "isaac_procedural_digit_plan_v0.1",
            "instruction":
                instruction,
            "planner_type":
                "procedural_font_to_points",
            "goal_type":
                "procedural_digit",
            "digit":
                digit,
            "approved":
                False,
            "step_count":
                0,
            "steps":
                [],
            "validation_errors": [
                "A single digit from 0 to 9 is required."
            ],
        }

    plan = build_procedural_letter_plan(
        instruction,
        digit,
    )

    plan["schema_version"] = (
        "isaac_procedural_digit_plan_v0.1"
    )
    plan["goal_type"] = "procedural_digit"
    plan["digit"] = digit
    plan.pop("letter", None)

    plan["goal_description"] = (
        f"Approximate digit {digit} using eight points "
        "sampled from a rendered font outline."
    )

    plan["approximation_note"] = (
        "The digit was generated procedurally from a system "
        "font. No digit-specific robot coordinates or "
        "number templates were used."
    )

    return plan
