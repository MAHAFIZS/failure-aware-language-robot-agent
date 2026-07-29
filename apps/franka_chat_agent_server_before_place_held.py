# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


parser = argparse.ArgumentParser(
    description="Persistent Isaac Sim chat-controlled Franka agent."
)

parser.add_argument(
    "--command-file",
    default="data/agent_command.json",
)

parser.add_argument(
    "--status-file",
    default="data/agent_status.json",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--headless",
    action="store_true",
)

parser.add_argument(
    "--max-action-steps",
    type=int,
    default=12000,
)

args, unknown = parser.parse_known_args()


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {"headless": bool(args.headless)}
)


import carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import (
    DynamicCuboid,
    VisualCuboid,
)
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import (
    PickPlaceController,
)
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

from workcell_environment import build_workcell


np.random.seed(args.seed)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMAND_FILE = Path(args.command_file).expanduser()
STATUS_FILE = Path(args.status_file).expanduser()

if not COMMAND_FILE.is_absolute():
    COMMAND_FILE = PROJECT_ROOT / COMMAND_FILE

if not STATUS_FILE.is_absolute():
    STATUS_FILE = PROJECT_ROOT / STATUS_FILE

COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

CUBE_SIZE = 0.0515
CUBE_Z = CUBE_SIZE / 2.0
SPAWN_Z = 0.30

SUCCESS_THRESHOLD = 0.05
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_OFFSET_M = 0.02

KNOWN_OBJECTS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "white",
}

WORKSPACE = {
    "x_min": 0.32,
    "x_max": 0.65,
    "y_min": -0.38,
    "y_max": 0.38,
    "z_min": CUBE_Z,
    "z_max": CUBE_Z + 7 * CUBE_SIZE,
}

MIN_TARGET_SPACING = 0.08
STACK_XY_TOLERANCE = 0.035
STACK_Z_TOLERANCE = 0.018
PLACEMENT_Z_THRESHOLD = 0.035
MAX_PLAN_STEPS = 16


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def write_status(
    *,
    state: str,
    command_id: str | None = None,
    **extra: Any,
) -> None:
    payload = {
        "state": state,
        "command_id": command_id,
        "updated_at_unix": time.time(),
        **extra,
    }

    write_json_atomic(STATUS_FILE, payload)


def distance_xy(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first[:2], dtype=float)
            - np.asarray(second[:2], dtype=float)
        )
    )


def read_command() -> dict[str, Any] | None:
    if not COMMAND_FILE.exists():
        return None

    try:
        return json.loads(
            COMMAND_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    steps = plan.get("steps")

    if not plan.get("approved"):
        errors.append(
            "The planner did not approve this plan."
        )

    if not isinstance(steps, list) or not steps:
        errors.append(
            "The plan contains no executable steps."
        )
        return errors

    if len(steps) > MAX_PLAN_STEPS:
        errors.append(
            f"Plan exceeds {MAX_PLAN_STEPS} steps."
        )

    used_objects: set[str] = set()
    targets: list[tuple[float, float]] = []

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(
                f"Step {index} is malformed."
            )
            continue

        if step.get("tool") not in {
            "move_object",
            "place_on_object",
            "pick_and_hold",
            "release_object",
        }:
            errors.append(
                f"Step {index} uses an unsupported tool."
            )

        # Release acts on the object already held by the robot,
        # so it does not require an object name in the plan.
        if step.get("tool") == "release_object":
            continue

        object_name = step.get("object")

        if object_name not in KNOWN_OBJECTS:
            errors.append(
                f"Step {index} references an unknown object."
            )

        if object_name in used_objects:
            errors.append(
                f"Step {index} repeats object {object_name}."
            )

        if object_name in KNOWN_OBJECTS:
            used_objects.add(object_name)

        if step.get("tool") in {
            "pick_and_hold",
            "release_object",
        }:
            continue

        target = step.get("target_position")

        if (
            not isinstance(target, list)
            or len(target) != 3
        ):
            errors.append(
                f"Step {index} has an invalid target."
            )
            continue

        try:
            x, y, z = map(float, target)
        except (TypeError, ValueError):
            errors.append(
                f"Step {index} target is not numeric."
            )
            continue

        if not WORKSPACE["x_min"] <= x <= WORKSPACE["x_max"]:
            errors.append(
                f"Step {index} X is outside the workspace."
            )

        if not WORKSPACE["y_min"] <= y <= WORKSPACE["y_max"]:
            errors.append(
                f"Step {index} Y is outside the workspace."
            )

        if not (
            WORKSPACE["z_min"]
            <= z
            <= WORKSPACE["z_max"]
        ):
            errors.append(
                f"Step {index} has an unsafe Z value."
            )

        # Every valid stacking height must align with
        # an integer cube level.
        stack_level = round(
            (z - CUBE_Z) / CUBE_SIZE
        )

        expected_z = (
            CUBE_Z
            + stack_level * CUBE_SIZE
        )

        if abs(z - expected_z) > STACK_Z_TOLERANCE:
            errors.append(
                f"Step {index} does not align with "
                "a valid cube stacking level."
            )

        targets.append((x, y, z))

    for first_index in range(len(targets)):
        for second_index in range(
            first_index + 1,
            len(targets),
        ):
            first_x, first_y, first_z = targets[first_index]
            second_x, second_y, second_z = targets[second_index]

            xy_distance = (
                (first_x - second_x) ** 2
                + (first_y - second_y) ** 2
            ) ** 0.5

            z_distance = abs(first_z - second_z)

            # Same XY is permitted when cubes occupy any two
            # different valid stacking levels, including levels
            # separated by more than one cube height.
            level_difference = round(
                z_distance / CUBE_SIZE
            )

            vertically_stacked = (
                xy_distance <= STACK_XY_TOLERANCE
                and level_difference >= 1
                and abs(
                    z_distance
                    - level_difference * CUBE_SIZE
                ) <= STACK_Z_TOLERANCE
            )

            if (
                xy_distance + 1e-6
                < MIN_TARGET_SPACING
                and not vertically_stacked
            ):
                errors.append(
                    "Two target positions are too close "
                    "and are not valid adjacent stack levels."
                )

    return errors


assets_root_path = get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not locate Isaac Sim assets.")
    simulation_app.close()
    raise SystemExit(1)


# ---------------------------------------------------------------------
# Persistent world
# ---------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

workcell_metadata = build_workcell(
    world,
    add_sorting_trays=False,
)

asset_path = (
    assets_root_path
    + "/Isaac/Robots/FrankaRobotics/"
    + "FrankaPanda/franka.usd"
)

robot_prim = add_reference_to_stage(
    usd_path=asset_path,
    prim_path="/World/Franka",
)

robot_prim.GetVariantSet(
    "Gripper"
).SetVariantSelection("AlternateFinger")

robot_prim.GetVariantSet(
    "Mesh"
).SetVariantSelection("Quality")

gripper = ParallelGripper(
    end_effector_prim_path=(
        "/World/Franka/panda_rightfinger"
    ),
    joint_prim_names=[
        "panda_finger_joint1",
        "panda_finger_joint2",
    ],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.02, 0.02]),
    action_deltas=np.array([0.01, 0.01]),
)

franka = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="franka",
        end_effector_prim_path=(
            "/World/Franka/panda_rightfinger"
        ),
        gripper=gripper,
    )
)


definitions = {
    "red": {
        "position": np.array([0.38, 0.30, SPAWN_Z]),
        "color": np.array([1.0, 0.0, 0.0]),
    },
    "green": {
        "position": np.array([0.49, 0.30, SPAWN_Z]),
        "color": np.array([0.0, 1.0, 0.0]),
    },
    "blue": {
        "position": np.array([0.60, 0.30, SPAWN_Z]),
        "color": np.array([0.0, 0.2, 1.0]),
    },
    "yellow": {
        "position": np.array([0.38, 0.10, SPAWN_Z]),
        "color": np.array([1.0, 0.85, 0.0]),
    },
    "orange": {
        "position": np.array([0.49, 0.10, SPAWN_Z]),
        "color": np.array([1.0, 0.35, 0.0]),
    },
    "purple": {
        "position": np.array([0.60, 0.10, SPAWN_Z]),
        "color": np.array([0.55, 0.10, 0.75]),
    },
    "cyan": {
        "position": np.array([0.41, -0.15, SPAWN_Z]),
        "color": np.array([0.0, 0.85, 0.85]),
    },
    "white": {
        "position": np.array([0.56, -0.15, SPAWN_Z]),
        "color": np.array([0.90, 0.90, 0.90]),
    },
}

cubes: dict[str, DynamicCuboid] = {}

for label, definition in definitions.items():
    cubes[label] = world.scene.add(
        DynamicCuboid(
            name=f"{label}_cube",
            prim_path=f"/World/{label.capitalize()}Cube",
            position=definition["position"],
            scale=np.array(
                [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]
            ),
            size=1.0,
            color=definition["color"],
        )
    )


franka.gripper.set_default_state(
    franka.gripper.joint_opened_positions
)

world.reset()

controller = PickPlaceController(
    name="persistent_chat_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)

articulation_controller = (
    franka.get_articulation_controller()
)


# ---------------------------------------------------------------------
# Persistent command state
# ---------------------------------------------------------------------

last_command_id: str | None = None
active_command_id: str | None = None
active_instruction = ""
active_plan: dict[str, Any] | None = None
active_steps: list[dict[str, Any]] = []

current_step_index = 0
attempt_index = 0
active_step: dict[str, Any] | None = None
active_cube: DynamicCuboid | None = None
active_target: np.ndarray | None = None

command_start_time = 0.0
command_action_steps = 0
step_results: list[dict[str, Any]] = []
target_marker_counter = 0

held_object_name: str | None = None
pick_initial_position: np.ndarray | None = None

initial_settle_steps = 180

write_status(
    state="starting",
    message="Isaac Sim is initializing.",
)

print("\n[Chat Agent] Persistent Isaac server started.")
print(f"[Chat Agent] Command file: {COMMAND_FILE}")
print(f"[Chat Agent] Status file:  {STATUS_FILE}")


def current_scene() -> dict[str, Any]:
    return {
        label: {
            "position": np.asarray(
                cube.get_local_pose()[0],
                dtype=float,
            ).tolist()
        }
        for label, cube in cubes.items()
    }


def complete_command(
    success: bool,
    *,
    keep_holding: bool = False,
) -> None:
    global active_command_id
    global active_instruction
    global active_plan
    global active_steps
    global current_step_index
    global attempt_index
    global active_step
    global active_cube
    global active_target
    global step_results
    global command_action_steps

    duration = time.time() - command_start_time

    result = {
        "schema_version":
            "isaac_persistent_chat_execution_v0.1",
        "holding_object":
            held_object_name if keep_holding else None,
        "command_id":
            active_command_id,
        "instruction":
            active_instruction,
        "success":
            bool(success),
        "requested_step_count":
            len(active_steps),
        "completed_step_count":
            len(step_results),
        "step_results":
            step_results,
        "final_scene":
            current_scene(),
        "duration_sec":
            duration,
        "simulation_action_steps":
            command_action_steps,
        "plan":
            active_plan,
        "environment":
            workcell_metadata,
    }

    result_path = (
        PROJECT_ROOT
        / "results"
        / "latest_persistent_chat_result.json"
    )

    write_json_atomic(result_path, result)

    write_status(
        state=(
            "holding"
            if success and keep_holding
            else "completed"
            if success
            else "failed"
        ),
        command_id=active_command_id,
        result=result,
        held_object=(
            held_object_name
            if keep_holding
            else None
        ),
    )

    print(
        f"\n[Chat Agent] Command completed: "
        f"success={success}"
    )

    active_command_id = None
    active_instruction = ""
    active_plan = None
    active_steps = []
    current_step_index = 0
    attempt_index = 0
    active_step = None
    active_cube = None
    active_target = None
    step_results = []
    command_action_steps = 0

    if not keep_holding:
        controller.reset()
        franka.gripper.open()


while simulation_app.is_running():
    world.step(render=not args.headless)

    if not world.is_playing():
        continue

    if initial_settle_steps > 0:
        initial_settle_steps -= 1

        if initial_settle_steps == 0:
            write_status(
                state="idle",
                message="Robot is ready for a command.",
                scene=current_scene(),
            )

            print("[Chat Agent] Robot is ready.")

        continue

    # -------------------------------------------------------------
    # Wait for a new command
    # -------------------------------------------------------------

    if active_command_id is None:
        command = read_command()

        if not command:
            continue

        command_id = str(
            command.get("command_id", "")
        )

        if (
            not command_id
            or command_id == last_command_id
        ):
            continue

        last_command_id = command_id

        if command.get("command_type") == "shutdown":
            write_status(
                state="shutting_down",
                command_id=command_id,
            )
            break

        plan = command.get("plan", {})
        validation_errors = validate_plan(plan)

        if validation_errors:
            write_status(
                state="rejected",
                command_id=command_id,
                instruction=command.get(
                    "instruction",
                    "",
                ),
                validation_errors=validation_errors,
                plan=plan,
            )

            print(
                "[Chat Agent] Command rejected:",
                validation_errors,
            )
            continue

        if (
            plan.get("steps")
            and plan["steps"][0].get("tool")
            == "release_object"
        ):
            if held_object_name is None:
                write_status(
                    state="rejected",
                    command_id=command_id,
                    validation_errors=[
                        "The robot is not holding an object."
                    ],
                    plan=plan,
                )
                continue

            franka.gripper.open()

            for _ in range(90):
                world.step(render=not args.headless)

            released_object = held_object_name
            held_object_name = None

            result = {
                "schema_version":
                    "isaac_persistent_chat_execution_v0.1",
                "command_id":
                    command_id,
                "instruction":
                    str(command.get("instruction", "")),
                "success":
                    True,
                "requested_step_count":
                    1,
                "completed_step_count":
                    1,
                "step_results": [
                    {
                        "tool": "release_object",
                        "object": released_object,
                        "success": True,
                    }
                ],
                "holding_object":
                    None,
                "final_scene":
                    current_scene(),
            }

            controller.reset()

            write_status(
                state="completed",
                command_id=command_id,
                result=result,
                held_object=None,
            )

            print(
                f"[Chat Agent] RELEASED "
                f"{released_object}"
            )
            continue

        active_command_id = command_id
        active_instruction = str(
            command.get("instruction", "")
        )
        active_plan = plan
        active_steps = list(plan["steps"])

        current_step_index = 0
        attempt_index = 0
        active_step = None
        active_cube = None
        active_target = None

        command_start_time = time.time()
        command_action_steps = 0
        step_results = []

        write_status(
            state="executing",
            command_id=active_command_id,
            instruction=active_instruction,
            current_step=0,
            total_steps=len(active_steps),
            plan=active_plan,
        )

        print(
            f"\n[Chat Agent] Executing command: "
            f"{active_instruction}"
        )

        continue

    # -------------------------------------------------------------
    # Execute an active command
    # -------------------------------------------------------------

    command_action_steps += 1

    if command_action_steps >= args.max_action_steps:
        step_results.append(
            {
                "step_index": current_step_index,
                "success": False,
                "failure_type": "maximum_steps_reached",
            }
        )

        complete_command(False)
        continue

    if active_step is None:
        if current_step_index >= len(active_steps):
            complete_command(True)
            continue

        active_step = active_steps[current_step_index]
        object_name = active_step["object"]

        active_cube = cubes[object_name]

        if active_step["tool"] == "pick_and_hold":
            active_target = np.asarray(
                active_cube.get_local_pose()[0],
                dtype=float,
            )

            pick_initial_position = active_target.copy()
        else:
            active_target = np.asarray(
                active_step["target_position"],
                dtype=float,
            )

            target_marker_counter += 1

            world.scene.add(
                VisualCuboid(
                    name=(
                        f"chat_target_marker_"
                        f"{target_marker_counter}"
                    ),
                    prim_path=(
                        f"/World/ChatTargetMarker"
                        f"{target_marker_counter}"
                    ),
                    position=np.array(
                        [
                            active_target[0],
                            active_target[1],
                            0.005,
                        ]
                    ),
                    scale=np.array([0.07, 0.07, 0.006]),
                    size=1.0,
                    color=np.array([1.0, 1.0, 0.0]),
                )
            )

        write_status(
            state="executing",
            command_id=active_command_id,
            instruction=active_instruction,
            current_step=current_step_index + 1,
            total_steps=len(active_steps),
            active_object=object_name,
            active_target=active_target.tolist(),
            plan=active_plan,
        )

        print(
            f"[Chat Agent] Step "
            f"{current_step_index + 1}/"
            f"{len(active_steps)}: "
            f"{object_name} → "
            f"{active_target.tolist()}"
        )

    assert active_cube is not None
    assert active_target is not None
    assert active_step is not None

    picking_position = np.asarray(
        active_cube.get_local_pose()[0],
        dtype=float,
    )

    actions = controller.forward(
        picking_position=picking_position,
        placing_position=active_target,
        current_joint_positions=(
            franka.get_joint_positions()
        ),
        end_effector_offset=np.array(
            [0.0, 0.005, 0.0]
        ),
    )

    articulation_controller.apply_action(actions)

    if (
        active_step["tool"] == "pick_and_hold"
        and controller.get_current_event() >= 5
    ):
        controller.pause()

        final_position = np.asarray(
            active_cube.get_local_pose()[0],
            dtype=float,
        )

        lift_distance = (
            float(final_position[2])
            - float(pick_initial_position[2])
            if pick_initial_position is not None
            else 0.0
        )

        holding_success = lift_distance >= 0.03

        step_results.append(
            {
                "step_index": current_step_index,
                "object": active_step["object"],
                "tool": "pick_and_hold",
                "initial_position":
                    pick_initial_position.tolist()
                    if pick_initial_position is not None
                    else None,
                "final_position":
                    final_position.tolist(),
                "lift_distance_m":
                    lift_distance,
                "success":
                    holding_success,
                "state":
                    "holding"
                    if holding_success
                    else "pick_failed",
            }
        )

        if holding_success:
            held_object_name = active_step["object"]

            print(
                f"[Chat Agent] HOLDING "
                f"{held_object_name}: "
                f"lift={lift_distance:.4f} m"
            )

            complete_command(
                True,
                keep_holding=True,
            )
        else:
            print(
                "[Chat Agent] Pick-and-hold failed: "
                f"lift={lift_distance:.4f} m"
            )

            complete_command(False)

        continue

    if controller.is_done():
        final_position = np.asarray(
            active_cube.get_local_pose()[0],
            dtype=float,
        )

        xy_error = distance_xy(
            final_position,
            active_target,
        )

        z_error = abs(
            float(final_position[2])
            - float(active_target[2])
        )

        error_3d = float(
            np.linalg.norm(
                final_position
                - active_target
            )
        )

        success = (
            xy_error <= SUCCESS_THRESHOLD
            and z_error <= PLACEMENT_Z_THRESHOLD
        )

        print(
            f"[Chat Agent] Verification: "
            f"xy_error={xy_error:.4f} m, "
            f"z_error={z_error:.4f} m, "
            f"success={success}"
        )

        if success:
            step_results.append(
                {
                    "step_index": current_step_index,
                    "object": active_step["object"],
                    "attempts": attempt_index + 1,
                    "target_position":
                        active_target.tolist(),
                    "final_position":
                        final_position.tolist(),
                    "placement_error_xy_m":
                        xy_error,
                    "placement_error_z_m":
                        z_error,
                    "placement_error_3d_m":
                        error_3d,
                    "success":
                        True,
                }
            )

            current_step_index += 1
            attempt_index = 0
            active_step = None
            active_cube = None
            active_target = None

            controller.reset()
            franka.gripper.open()

        elif attempt_index + 1 >= MAX_RECOVERY_ATTEMPTS:
            step_results.append(
                {
                    "step_index": current_step_index,
                    "object": active_step["object"],
                    "attempts": attempt_index + 1,
                    "target_position":
                        active_target.tolist(),
                    "final_position":
                        final_position.tolist(),
                    "placement_error_xy_m":
                        xy_error,
                    "placement_error_z_m":
                        z_error,
                    "placement_error_3d_m":
                        error_3d,
                    "success":
                        False,
                    "failure_type":
                        "recovery_exhausted",
                }
            )

            complete_command(False)

        else:
            attempt_index += 1

            if attempt_index == 1:
                active_target[0] += RECOVERY_OFFSET_M
            else:
                active_target[1] -= RECOVERY_OFFSET_M

            print(
                f"[Chat Agent] Recovery attempt "
                f"{attempt_index + 1}/"
                f"{MAX_RECOVERY_ATTEMPTS}"
            )

            controller.reset()
            franka.gripper.open()


write_status(
    state="stopped",
    message="Isaac persistent server stopped.",
)

simulation_app.close()
