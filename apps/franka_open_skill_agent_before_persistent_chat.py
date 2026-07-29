# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Persistent open-skill Franka agent."
)

parser.add_argument(
    "--plan-json",
    required=True,
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
    "--result-json",
    default="results/latest_open_skill_execution.json",
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=30000,
)

args, unknown = parser.parse_known_args()


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {"headless": bool(args.headless)}
)


import carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import (
    PickPlaceController,
)
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

from workcell_environment import build_workcell


np.random.seed(args.seed)

CUBE_SIZE = 0.0515
CUBE_Z = CUBE_SIZE / 2.0
SPAWN_Z = 0.30

SUCCESS_THRESHOLD = 0.05
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_OFFSET_M = 0.02

PROJECT_ROOT = Path(__file__).resolve().parents[1]

plan_path = Path(args.plan_json).expanduser()

if not plan_path.is_absolute():
    plan_path = PROJECT_ROOT / plan_path

result_path = Path(args.result_json).expanduser()

if not result_path.is_absolute():
    result_path = PROJECT_ROOT / result_path


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


if not plan_path.exists():
    simulation_app.close()
    raise FileNotFoundError(
        f"Plan not found: {plan_path}"
    )

plan = json.loads(
    plan_path.read_text(encoding="utf-8")
)

if not plan.get("approved"):
    simulation_app.close()
    raise ValueError("Plan was not approved.")

steps = plan.get("steps", [])

if not steps:
    simulation_app.close()
    raise ValueError("Plan contains no steps.")


assets_root_path = get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not locate Isaac Sim assets.")
    simulation_app.close()
    raise SystemExit(1)


# ---------------------------------------------------------------------
# World and robot
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


# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------

definitions = {
    "red": {
        "position": np.array([0.30, 0.30, SPAWN_Z]),
        "color": np.array([1.0, 0.0, 0.0]),
    },
    "green": {
        "position": np.array([0.42, 0.30, SPAWN_Z]),
        "color": np.array([0.0, 1.0, 0.0]),
    },
    "blue": {
        "position": np.array([0.54, 0.30, SPAWN_Z]),
        "color": np.array([0.0, 0.2, 1.0]),
    },
    "yellow": {
        "position": np.array([0.30, 0.10, SPAWN_Z]),
        "color": np.array([1.0, 0.85, 0.0]),
    },
    "orange": {
        "position": np.array([0.42, 0.10, SPAWN_Z]),
        "color": np.array([1.0, 0.35, 0.0]),
    },
    "purple": {
        "position": np.array([0.54, 0.10, SPAWN_Z]),
        "color": np.array([0.55, 0.1, 0.75]),
    },
    "cyan": {
        "position": np.array([0.34, -0.15, SPAWN_Z]),
        "color": np.array([0.0, 0.85, 0.85]),
    },
    "white": {
        "position": np.array([0.50, -0.15, SPAWN_Z]),
        "color": np.array([0.9, 0.9, 0.9]),
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
    name="open_skill_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)

articulation_controller = (
    franka.get_articulation_controller()
)


# ---------------------------------------------------------------------
# Persistent multi-step execution
# ---------------------------------------------------------------------

current_step_index = 0
attempt_index = 0
step_results: list[dict] = []

step_count = 0
initial_settle_steps = 120
completion_pause_steps = 120
finished = False

start_time = time.time()

active_target = None
active_cube = None
active_step = None


print("\n[Open Skill Agent] Plan:")
print(json.dumps(plan, indent=2))


while simulation_app.is_running():
    world.step(render=not args.headless)
    step_count += 1

    if step_count >= args.max_steps:
        print("[Open Skill Agent] Maximum steps reached.")
        break

    if not world.is_playing():
        continue

    if initial_settle_steps > 0:
        initial_settle_steps -= 1
        continue

    if finished:
        completion_pause_steps -= 1

        if completion_pause_steps <= 0:
            break

        continue

    if active_step is None:
        if current_step_index >= len(steps):
            finished = True
            continue

        active_step = steps[current_step_index]

        if active_step.get("tool") != "move_object":
            step_results.append(
                {
                    "step_index": current_step_index,
                    "success": False,
                    "failure_type": "unsupported_tool",
                    "step": active_step,
                }
            )
            finished = True
            continue

        object_name = active_step["object"]
        active_cube = cubes[object_name]

        active_target = np.asarray(
            active_step["target_position"],
            dtype=float,
        )

        world.scene.add(
            VisualCuboid(
                name=f"target_marker_{current_step_index}",
                prim_path=(
                    f"/World/TargetMarker{current_step_index}"
                ),
                position=np.array(
                    [
                        active_target[0],
                        active_target[1],
                        0.005,
                    ]
                ),
                scale=np.array([0.09, 0.09, 0.01]),
                size=1.0,
                color=np.array([1.0, 1.0, 0.0]),
            )
        )

        print(
            f"\n[Open Skill Agent] Step "
            f"{current_step_index + 1}/{len(steps)}"
        )
        print(json.dumps(active_step, indent=2))

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

    if controller.is_done():
        final_position = np.asarray(
            active_cube.get_local_pose()[0],
            dtype=float,
        )

        error = distance_xy(
            final_position,
            active_target,
        )

        success = error <= SUCCESS_THRESHOLD

        print(
            f"[Open Skill Agent] Verification: "
            f"error={error:.4f} m, "
            f"success={success}"
        )

        if success:
            step_results.append(
                {
                    "step_index": current_step_index,
                    "attempts": attempt_index + 1,
                    "object": active_step["object"],
                    "target_position":
                        active_target.tolist(),
                    "final_position":
                        final_position.tolist(),
                    "placement_error_xy_m":
                        error,
                    "success":
                        True,
                }
            )

            current_step_index += 1
            attempt_index = 0
            active_step = None
            active_target = None
            active_cube = None

            controller.reset()
            franka.gripper.open()

        elif attempt_index + 1 >= MAX_RECOVERY_ATTEMPTS:
            step_results.append(
                {
                    "step_index": current_step_index,
                    "attempts": attempt_index + 1,
                    "object": active_step["object"],
                    "target_position":
                        active_target.tolist(),
                    "final_position":
                        final_position.tolist(),
                    "placement_error_xy_m":
                        error,
                    "success":
                        False,
                    "failure_type":
                        "recovery_exhausted",
                }
            )

            finished = True

        else:
            attempt_index += 1

            if attempt_index == 1:
                active_target[0] += RECOVERY_OFFSET_M
            else:
                active_target[1] -= RECOVERY_OFFSET_M

            print(
                f"[Open Skill Agent] Recovery attempt "
                f"{attempt_index + 1}"
            )

            controller.reset()
            franka.gripper.open()


duration_sec = time.time() - start_time

all_success = (
    len(step_results) == len(steps)
    and all(
        bool(item.get("success"))
        for item in step_results
    )
)

final_scene = {
    label: {
        "position": np.asarray(
            cube.get_local_pose()[0],
            dtype=float,
        ).tolist()
    }
    for label, cube in cubes.items()
}


result = {
    "schema_version":
        "isaac_open_skill_execution_v0.1",
    "simulator":
        "isaac_sim",
    "robot":
        "franka_panda",
    "instruction":
        plan.get("instruction"),
    "goal_type":
        plan.get("goal_type"),
    "planner_type":
        plan.get("planner_type"),
    "plan":
        plan,
    "requested_step_count":
        len(steps),
    "completed_step_count":
        len(step_results),
    "success":
        all_success,
    "failure_type":
        None if all_success else "open_skill_execution_failed",
    "step_results":
        step_results,
    "final_scene":
        final_scene,
    "simulation_steps":
        step_count,
    "duration_sec":
        duration_sec,
    "environment":
        workcell_metadata,
    "autonomy_features": {
        "persistent_scene":
            True,
        "compositional_planning":
            True,
        "primitive_skill_execution":
            True,
        "post_action_verification":
            True,
        "failure_recovery":
            True,
        "new_whole_task_script_required":
            False,
    },
}

result_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_path.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("\n[Open Skill Agent] Final result:")
print(json.dumps(result, indent=2))

print(
    f"\n[Open Skill Agent] Result saved: "
    f"{result_path}"
)

simulation_app.close()
