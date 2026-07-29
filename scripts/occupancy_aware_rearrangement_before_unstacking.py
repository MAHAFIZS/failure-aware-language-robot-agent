from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


OCCUPANCY_TOLERANCE_M = 0.045
BUFFER_CLEARANCE_M = 0.085
TABLE_Z_M = 0.02575

WORKSPACE = {
    "x_min": 0.32,
    "x_max": 0.65,
    "y_min": -0.38,
    "y_max": 0.38,
}


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


def find_occupier(
    positions: dict[str, list[float]],
    target: list[float],
    moving_object: str,
) -> str | None:
    for object_name, position in positions.items():
        if object_name == moving_object:
            continue

        if xy_distance(position, target) <= OCCUPANCY_TOLERANCE_M:
            return object_name

    return None


def choose_free_buffer(
    positions: dict[str, list[float]],
    final_targets: list[list[float]],
    moving_object: str,
) -> list[float]:
    candidates = [
        [0.34, -0.34, TABLE_Z_M],
        [0.45, -0.34, TABLE_Z_M],
        [0.56, -0.34, TABLE_Z_M],
        [0.64, -0.34, TABLE_Z_M],
        [0.34, 0.34, TABLE_Z_M],
        [0.45, 0.34, TABLE_Z_M],
        [0.56, 0.34, TABLE_Z_M],
        [0.64, 0.34, TABLE_Z_M],
        [0.34, -0.22, TABLE_Z_M],
        [0.64, -0.22, TABLE_Z_M],
        [0.34, 0.22, TABLE_Z_M],
        [0.64, 0.22, TABLE_Z_M],
    ]

    for candidate in candidates:
        if not (
            WORKSPACE["x_min"]
            <= candidate[0]
            <= WORKSPACE["x_max"]
        ):
            continue

        if not (
            WORKSPACE["y_min"]
            <= candidate[1]
            <= WORKSPACE["y_max"]
        ):
            continue

        too_close_to_object = any(
            object_name != moving_object
            and xy_distance(position, candidate)
            < BUFFER_CLEARANCE_M
            for object_name, position in positions.items()
        )

        if too_close_to_object:
            continue

        too_close_to_target = any(
            xy_distance(target, candidate)
            < BUFFER_CLEARANCE_M
            for target in final_targets
        )

        if too_close_to_target:
            continue

        return candidate

    raise RuntimeError(
        "No safe temporary buffer position is available."
    )


def expand_occupancy_aware_plan(
    positions: dict[str, list[float]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Reorder a flat geometric arrangement so the robot never
    places a cube onto an already occupied target.

    Cycles are resolved by moving one blocking cube to a
    temporary buffer location.
    """

    original_steps = plan.get("steps", [])

    if not original_steps:
        return plan

    if any(
        step.get("tool") != "move_object"
        for step in original_steps
    ):
        return plan

    expanded_plan = deepcopy(plan)

    step_by_object = {
        step["object"]: deepcopy(step)
        for step in original_steps
    }

    pending = [
        step["object"]
        for step in original_steps
    ]

    current_positions = {
        name: list(map(float, position))
        for name, position in positions.items()
    }

    final_targets = [
        list(map(float, step["target_position"]))
        for step in original_steps
    ]

    expanded_steps: list[dict[str, Any]] = []
    temporary_moves = 0
    safety_counter = 0

    while pending:
        safety_counter += 1

        if safety_counter > 100:
            raise RuntimeError(
                "Occupancy-aware planning did not converge."
            )

        made_progress = False

        # First execute any move whose destination is currently free.
        for object_name in list(pending):
            final_step = step_by_object[object_name]
            target = final_step["target_position"]

            occupier = find_occupier(
                current_positions,
                target,
                object_name,
            )

            if occupier is not None:
                continue

            expanded_steps.append(final_step)

            current_positions[object_name] = list(
                map(float, target)
            )

            pending.remove(object_name)
            made_progress = True

        if made_progress:
            continue

        # A dependency cycle or external blocker remains.
        requested_object = pending[0]
        requested_target = step_by_object[
            requested_object
        ]["target_position"]

        blocker = find_occupier(
            current_positions,
            requested_target,
            requested_object,
        )

        if blocker is None:
            raise RuntimeError(
                "No blocker was found although no move was possible."
            )

        buffer_position = choose_free_buffer(
            current_positions,
            final_targets,
            blocker,
        )

        expanded_steps.append(
            {
                "tool": "move_object",
                "object": blocker,
                "target_position": buffer_position,
                "reason": (
                    f"Temporarily move {blocker} because it "
                    f"occupies the target required by "
                    f"{requested_object}."
                ),
                "temporary_buffer_move": True,
            }
        )

        current_positions[blocker] = buffer_position
        temporary_moves += 1

    expanded_plan["steps"] = expanded_steps
    expanded_plan["step_count"] = len(expanded_steps)
    expanded_plan["occupancy_aware"] = True
    expanded_plan["temporary_move_count"] = temporary_moves
    expanded_plan["original_step_count"] = len(original_steps)

    note = expanded_plan.get(
        "approximation_note",
        "",
    )

    occupancy_note = (
        " Execution order was expanded using the current "
        "Isaac scene so occupied targets are cleared first."
    )

    expanded_plan["approximation_note"] = (
        note + occupancy_note
    ).strip()

    return expanded_plan
