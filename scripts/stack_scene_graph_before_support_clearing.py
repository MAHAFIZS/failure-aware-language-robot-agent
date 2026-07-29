from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CUBE_SIZE_M = 0.0515
TABLE_Z_M = CUBE_SIZE_M / 2.0

XY_STACK_TOLERANCE_M = 0.035
Z_STACK_TOLERANCE_M = 0.018

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


@dataclass
class StackNode:
    name: str
    position: list[float]
    support: str
    directly_above: list[str]


def xy_distance(
    first: list[float],
    second: list[float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first[:2], dtype=float)
            - np.asarray(second[:2], dtype=float)
        )
    )


def infer_scene_graph(
    positions: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    """
    Infer which cube supports each cube from measured XYZ positions.
    """

    graph: dict[str, dict[str, Any]] = {}

    for object_name, position in positions.items():
        graph[object_name] = {
            "position": list(map(float, position)),
            "support": "table",
            "directly_above": [],
        }

    for upper_name, upper_position in positions.items():
        best_support = "table"
        best_z_error = float("inf")

        for lower_name, lower_position in positions.items():
            if upper_name == lower_name:
                continue

            horizontal_error = xy_distance(
                upper_position,
                lower_position,
            )

            expected_upper_z = (
                float(lower_position[2])
                + CUBE_SIZE_M
            )

            vertical_error = abs(
                float(upper_position[2])
                - expected_upper_z
            )

            if (
                horizontal_error
                <= XY_STACK_TOLERANCE_M
                and vertical_error
                <= Z_STACK_TOLERANCE_M
                and vertical_error < best_z_error
            ):
                best_support = lower_name
                best_z_error = vertical_error

        graph[upper_name]["support"] = best_support

    for object_name, node in graph.items():
        support = node["support"]

        if support != "table" and support in graph:
            graph[support]["directly_above"].append(
                object_name
            )

    return graph


def find_all_objects_above(
    graph: dict[str, dict[str, Any]],
    object_name: str,
) -> list[str]:
    """
    Return blockers in top-to-bottom removal order.
    """

    ordered: list[str] = []

    def visit(current: str) -> None:
        for upper in graph[current]["directly_above"]:
            visit(upper)
            ordered.append(upper)

    visit(object_name)

    return ordered


def stack_chain_from_bottom(
    graph: dict[str, dict[str, Any]],
    bottom_object: str,
) -> list[str]:
    chain = [bottom_object]
    current = bottom_object

    while graph[current]["directly_above"]:
        next_object = graph[current]["directly_above"][0]
        chain.append(next_object)
        current = next_object

    return chain


def calculate_stack_target(
    base_position: list[float],
    level: int,
) -> list[float]:
    """
    level=0 means table.
    level=1 means on one cube.
    level=2 means on two cubes.
    """

    return [
        float(base_position[0]),
        float(base_position[1]),
        TABLE_Z_M + level * CUBE_SIZE_M,
    ]


def choose_buffer_positions(
    occupied_positions: dict[str, list[float]],
    count: int,
) -> list[list[float]]:
    candidates = [
        [0.35, -0.32, TABLE_Z_M],
        [0.47, -0.32, TABLE_Z_M],
        [0.59, -0.32, TABLE_Z_M],
        [0.35, -0.18, TABLE_Z_M],
        [0.47, -0.18, TABLE_Z_M],
        [0.59, -0.18, TABLE_Z_M],
    ]

    selected: list[list[float]] = []

    for candidate in candidates:
        safe = True

        for position in occupied_positions.values():
            if xy_distance(candidate, position) < 0.09:
                safe = False
                break

        for position in selected:
            if xy_distance(candidate, position) < 0.09:
                safe = False
                break

        if safe:
            selected.append(candidate)

        if len(selected) == count:
            return selected

    raise RuntimeError(
        "Not enough safe buffer positions are available."
    )


def plan_pick_with_unstacking(
    positions: dict[str, list[float]],
    requested_object: str,
) -> dict[str, Any]:
    if requested_object not in positions:
        return {
            "approved": False,
            "validation_errors": [
                f"Unknown object: {requested_object}"
            ],
            "steps": [],
        }

    graph = infer_scene_graph(positions)

    blockers = find_all_objects_above(
        graph,
        requested_object,
    )

    buffer_positions = choose_buffer_positions(
        positions,
        len(blockers),
    )

    steps: list[dict[str, Any]] = []

    for blocker, target in zip(
        blockers,
        buffer_positions,
    ):
        steps.append(
            {
                "tool": "move_object",
                "object": blocker,
                "target_position": target,
                "reason": (
                    f"Remove {blocker} because it blocks "
                    f"{requested_object}."
                ),
            }
        )

    steps.append(
        {
            "tool": "pick_and_hold",
            "object": requested_object,
            "reason": (
                f"Requested object {requested_object} is clear."
            ),
        }
    )

    return {
        "approved": True,
        "intent": "pick_object",
        "requested_object": requested_object,
        "blockers": blockers,
        "scene_graph": graph,
        "steps": steps,
        "validation_errors": [],
    }


def plan_place_on_object(
    positions: dict[str, list[float]],
    moving_object: str,
    support_object: str,
) -> dict[str, Any]:
    if moving_object == support_object:
        return {
            "approved": False,
            "validation_errors": [
                "An object cannot be placed on itself."
            ],
            "steps": [],
        }

    if moving_object not in positions:
        return {
            "approved": False,
            "validation_errors": [
                f"Unknown moving object: {moving_object}"
            ],
            "steps": [],
        }

    if support_object not in positions:
        return {
            "approved": False,
            "validation_errors": [
                f"Unknown support object: {support_object}"
            ],
            "steps": [],
        }

    graph = infer_scene_graph(positions)

    moving_blockers = find_all_objects_above(
        graph,
        moving_object,
    )

    support_chain = stack_chain_from_bottom(
        graph,
        support_object,
    )

    support_top = support_chain[-1]

    if support_top != support_object:
        return {
            "approved": False,
            "validation_errors": [
                (
                    f"{support_object} is not clear. "
                    f"{support_top} is above it."
                )
            ],
            "steps": [],
            "scene_graph": graph,
        }

    buffer_positions = choose_buffer_positions(
        positions,
        len(moving_blockers),
    )

    steps: list[dict[str, Any]] = []

    for blocker, target in zip(
        moving_blockers,
        buffer_positions,
    ):
        steps.append(
            {
                "tool": "move_object",
                "object": blocker,
                "target_position": target,
                "reason": (
                    f"Clear {moving_object} before moving it."
                ),
            }
        )

    support_position = positions[support_object]

    target_position = [
        float(support_position[0]),
        float(support_position[1]),
        float(support_position[2]) + CUBE_SIZE_M,
    ]

    steps.append(
        {
            "tool": "place_on_object",
            "object": moving_object,
            "support_object": support_object,
            "target_position": target_position,
        }
    )

    return {
        "approved": True,
        "intent": "place_on_object",
        "moving_object": moving_object,
        "support_object": support_object,
        "scene_graph": graph,
        "steps": steps,
        "validation_errors": [],
    }


def plan_tower(
    positions: dict[str, list[float]],
    bottom_to_top: list[str],
    tower_xy: list[float],
) -> dict[str, Any]:
    if len(bottom_to_top) < 2:
        return {
            "approved": False,
            "validation_errors": [
                "A tower requires at least two objects."
            ],
            "steps": [],
        }

    if len(set(bottom_to_top)) != len(bottom_to_top):
        return {
            "approved": False,
            "validation_errors": [
                "Tower objects must be unique."
            ],
            "steps": [],
        }

    unknown = [
        name
        for name in bottom_to_top
        if name not in positions
    ]

    if unknown:
        return {
            "approved": False,
            "validation_errors": [
                f"Unknown objects: {unknown}"
            ],
            "steps": [],
        }

    x, y = map(float, tower_xy)

    steps: list[dict[str, Any]] = []

    for level, object_name in enumerate(bottom_to_top):
        steps.append(
            {
                "tool": "move_object",
                "object": object_name,
                "target_position": [
                    x,
                    y,
                    TABLE_Z_M + level * CUBE_SIZE_M,
                ],
                "stack_level": level,
            }
        )

    return {
        "approved": True,
        "intent": "make_tower",
        "tower_order_bottom_to_top": bottom_to_top,
        "steps": steps,
        "validation_errors": [],
    }
