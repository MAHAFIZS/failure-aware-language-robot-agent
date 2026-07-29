from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


OCCUPANCY_TOLERANCE_M = 0.045
STACK_XY_TOLERANCE_M = 0.035
STACK_Z_MINIMUM_M = 0.025
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
    candidates: list[tuple[str, float]] = []

    for object_name, position in positions.items():
        if object_name == moving_object:
            continue

        if xy_distance(position, target) <= OCCUPANCY_TOLERANCE_M:
            candidates.append(
                (
                    object_name,
                    float(position[2]),
                )
            )

    if not candidates:
        return None

    # Topmost cube must be removed first.
    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates[0][0]


def find_objects_above(
    positions: dict[str, list[float]],
    object_name: str,
) -> list[str]:
    """
    Return cubes above object_name in top-to-bottom order.
    """

    if object_name not in positions:
        return []

    base_position = positions[object_name]

    blockers: list[tuple[str, float]] = []

    for candidate_name, candidate_position in positions.items():
        if candidate_name == object_name:
            continue

        same_stack = (
            xy_distance(
                candidate_position,
                base_position,
            )
            <= STACK_XY_TOLERANCE_M
        )

        clearly_above = (
            float(candidate_position[2])
            > float(base_position[2])
            + STACK_Z_MINIMUM_M
        )

        if same_stack and clearly_above:
            blockers.append(
                (
                    candidate_name,
                    float(candidate_position[2]),
                )
            )

    blockers.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        object_name
        for object_name, _ in blockers
    ]


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
        inside_workspace = (
            WORKSPACE["x_min"]
            <= candidate[0]
            <= WORKSPACE["x_max"]
            and WORKSPACE["y_min"]
            <= candidate[1]
            <= WORKSPACE["y_max"]
        )

        if not inside_workspace:
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


def make_temporary_move(
    blocker: str,
    buffer_position: list[float],
    reason: str,
) -> dict[str, Any]:
    return {
        "tool": "move_object",
        "object": blocker,
        "target_position": buffer_position,
        "reason": reason,
        "temporary_buffer_move": True,
    }


def expand_occupancy_aware_plan(
    positions: dict[str, list[float]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Expand a flat arrangement plan using the current scene.

    It handles:
    1. cubes stacked above the cube that must be moved;
    2. cubes occupying the requested destination;
    3. rearrangement cycles requiring temporary buffers.
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
    source_unstack_moves = 0
    destination_clear_moves = 0
    safety_counter = 0

    while pending:
        safety_counter += 1

        if safety_counter > 200:
            raise RuntimeError(
                "Occupancy-aware planning did not converge."
            )

        made_progress = False

        for object_name in list(pending):
            final_step = step_by_object[object_name]
            target = final_step["target_position"]

            source_blockers = find_objects_above(
                current_positions,
                object_name,
            )

            if source_blockers:
                continue

            destination_occupier = find_occupier(
                current_positions,
                target,
                object_name,
            )

            if destination_occupier is not None:
                continue

            expanded_steps.append(final_step)

            current_positions[object_name] = list(
                map(float, target)
            )

            pending.remove(object_name)
            made_progress = True

        if made_progress:
            continue

        requested_object = pending[0]
        requested_target = step_by_object[
            requested_object
        ]["target_position"]

        # First clear cubes stacked above the source.
        source_blockers = find_objects_above(
            current_positions,
            requested_object,
        )

        if source_blockers:
            blocker = source_blockers[0]

            buffer_position = choose_free_buffer(
                current_positions,
                final_targets,
                blocker,
            )

            expanded_steps.append(
                make_temporary_move(
                    blocker,
                    buffer_position,
                    (
                        f"Remove {blocker} because it is stacked "
                        f"above source cube {requested_object}."
                    ),
                )
            )

            current_positions[blocker] = buffer_position

            temporary_moves += 1
            source_unstack_moves += 1
            continue

        # Then clear the requested destination.
        destination_occupier = find_occupier(
            current_positions,
            requested_target,
            requested_object,
        )

        if destination_occupier is None:
            raise RuntimeError(
                "No blocker was found although no move was possible."
            )

        # If the destination occupier has cubes above it, remove
        # the topmost cube before attempting to move the occupier.
        occupier_blockers = find_objects_above(
            current_positions,
            destination_occupier,
        )

        if occupier_blockers:
            blocker = occupier_blockers[0]
            reason = (
                f"Remove {blocker} because it is stacked above "
                f"destination blocker {destination_occupier}."
            )
        else:
            blocker = destination_occupier
            reason = (
                f"Temporarily move {blocker} because it occupies "
                f"the target required by {requested_object}."
            )

        buffer_position = choose_free_buffer(
            current_positions,
            final_targets,
            blocker,
        )

        expanded_steps.append(
            make_temporary_move(
                blocker,
                buffer_position,
                reason,
            )
        )

        current_positions[blocker] = buffer_position

        temporary_moves += 1
        destination_clear_moves += 1

    expanded_plan["steps"] = expanded_steps
    expanded_plan["step_count"] = len(expanded_steps)
    expanded_plan["occupancy_aware"] = True
    expanded_plan["stack_aware"] = True
    expanded_plan["temporary_move_count"] = temporary_moves
    expanded_plan["source_unstack_move_count"] = (
        source_unstack_moves
    )
    expanded_plan["destination_clear_move_count"] = (
        destination_clear_moves
    )
    expanded_plan["original_step_count"] = len(
        original_steps
    )

    note = expanded_plan.get(
        "approximation_note",
        "",
    )

    added_note = (
        " Execution order was expanded using the current "
        "Isaac scene. Occupied destinations and cubes stacked "
        "above source objects are cleared before final placement."
    )

    expanded_plan["approximation_note"] = (
        note + added_note
    ).strip()

    return expanded_plan
