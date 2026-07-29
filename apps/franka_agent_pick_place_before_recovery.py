# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Scene-aware Franka pick-and-place agent."
)

parser.add_argument(
    "--instruction",
    default="Pick the red box and place it beside the blue box.",
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--headless", action="store_true")

parser.add_argument(
    "--result-json",
    default="results/latest_isaac_agent_result.json",
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=15000,
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
RELATIVE_DISTANCE = 0.12
SUCCESS_THRESHOLD = 0.08

PROJECT_ROOT = Path(__file__).resolve().parents[1]

result_path = Path(args.result_json).expanduser()

if not result_path.is_absolute():
    result_path = PROJECT_ROOT / result_path


def parse_instruction(instruction: str) -> dict:
    text = instruction.lower().strip()

    colors = re.findall(
        r"\b(red|green|blue)\b",
        text,
    )

    if len(colors) < 2:
        raise ValueError(
            "Instruction must contain an object color and "
            "a reference-object color."
        )

    moving_object = colors[0]
    reference_object = colors[1]

    if "left of" in text:
        relation = "left_of"
    elif "right of" in text:
        relation = "right_of"
    elif "in front of" in text or "front of" in text:
        relation = "in_front_of"
    elif "behind" in text:
        relation = "behind"
    elif "beside" in text or "next to" in text:
        relation = "beside"
    else:
        relation = "beside"

    return {
        "action": "pick_place_relative",
        "moving_object": moving_object,
        "reference_object": reference_object,
        "relation": relation,
    }


def calculate_relative_target(
    reference_position: np.ndarray,
    relation: str,
) -> np.ndarray:
    offsets = {
        "left_of": np.array(
            [0.0, RELATIVE_DISTANCE, 0.0]
        ),
        "right_of": np.array(
            [0.0, -RELATIVE_DISTANCE, 0.0]
        ),
        "in_front_of": np.array(
            [RELATIVE_DISTANCE, 0.0, 0.0]
        ),
        "behind": np.array(
            [-RELATIVE_DISTANCE, 0.0, 0.0]
        ),
        "beside": np.array(
            [0.0, RELATIVE_DISTANCE, 0.0]
        ),
    }

    target = (
        np.asarray(reference_position, dtype=float)
        + offsets[relation]
    )

    target[2] = CUBE_Z
    return target


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


try:
    plan = parse_instruction(args.instruction)
except ValueError as exc:
    print(f"[Agent] {exc}")
    simulation_app.close()
    raise SystemExit(1)


assets_root_path = get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not locate Isaac Sim assets.")
    simulation_app.close()
    raise SystemExit(1)


# ---------------------------------------------------------------------
# World and workcell
# ---------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

workcell_metadata = build_workcell(
    world,
    add_sorting_trays=False,
)


# ---------------------------------------------------------------------
# Franka
# ---------------------------------------------------------------------

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
# Scene objects
# ---------------------------------------------------------------------

object_definitions = {
    "red": {
        "position": np.array(
            [0.35, 0.28, SPAWN_Z]
        ),
        "color": np.array([1.0, 0.0, 0.0]),
    },
    "green": {
        "position": np.array(
            [0.45, 0.00, SPAWN_Z]
        ),
        "color": np.array([0.0, 1.0, 0.0]),
    },
    "blue": {
        "position": np.array(
            [0.35, -0.25, SPAWN_Z]
        ),
        "color": np.array([0.0, 0.2, 1.0]),
    },
}

cubes: dict[str, DynamicCuboid] = {}

for label, definition in object_definitions.items():
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
    name="agent_pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)

articulation_controller = (
    franka.get_articulation_controller()
)


moving_label = plan["moving_object"]
reference_label = plan["reference_object"]

moving_cube = cubes[moving_label]
reference_cube = cubes[reference_label]


# ---------------------------------------------------------------------
# Autonomous execution loop
# ---------------------------------------------------------------------

step_count = 0
initial_settle_steps = 120
completion_pause_steps = 120
finished = False

start_time = time.time()

initial_scene = {}
target_position = None
final_position = None
placement_error = None
success = False


print("\n[Agent] User instruction:")
print(args.instruction)

print("\n[Agent] Parsed plan:")
print(json.dumps(plan, indent=2))


while simulation_app.is_running():
    world.step(render=not args.headless)
    step_count += 1

    if step_count >= args.max_steps:
        print("[Agent] Maximum step count reached.")
        break

    if not world.is_playing():
        continue

    if initial_settle_steps > 0:
        initial_settle_steps -= 1
        continue

    if not initial_scene:
        initial_scene = {
            label: {
                "position": np.asarray(
                    cube.get_local_pose()[0],
                    dtype=float,
                ).tolist()
            }
            for label, cube in cubes.items()
        }

        reference_position = np.asarray(
            reference_cube.get_local_pose()[0],
            dtype=float,
        )

        target_position = calculate_relative_target(
            reference_position,
            plan["relation"],
        )

        world.scene.add(
            VisualCuboid(
                name="agent_target_marker",
                prim_path="/World/AgentTargetMarker",
                position=np.array(
                    [
                        target_position[0],
                        target_position[1],
                        0.005,
                    ]
                ),
                scale=np.array([0.10, 0.10, 0.01]),
                size=1.0,
                color=np.array([1.0, 1.0, 0.0]),
            )
        )

        print("\n[Agent] Scene inspection:")
        print(json.dumps(initial_scene, indent=2))

        print(
            "\n[Agent] Calculated target:",
            target_position.tolist(),
        )

    if finished:
        completion_pause_steps -= 1

        if completion_pause_steps <= 0:
            break

        continue

    picking_position = np.asarray(
        moving_cube.get_local_pose()[0],
        dtype=float,
    )

    actions = controller.forward(
        picking_position=picking_position,
        placing_position=target_position,
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
            moving_cube.get_local_pose()[0],
            dtype=float,
        )

        placement_error = distance_xy(
            final_position,
            target_position,
        )

        success = (
            placement_error <= SUCCESS_THRESHOLD
        )

        print(
            f"\n[Agent] Verification: "
            f"error={placement_error:.4f} m, "
            f"success={success}"
        )

        finished = True


duration_sec = time.time() - start_time

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
        "isaac_autonomous_agent_v0.1",
    "simulator":
        "isaac_sim",
    "robot":
        "franka_panda",
    "instruction":
        args.instruction,
    "seed":
        int(args.seed),
    "agent_type":
        "constrained_scene_aware_manipulation_agent",
    "planner":
        "rule_based_language_parser_and_relative_planner",
    "plan":
        plan,
    "initial_scene":
        initial_scene,
    "calculated_target_position":
        (
            target_position.tolist()
            if target_position is not None
            else None
        ),
    "final_scene":
        final_scene,
    "final_object_position":
        (
            final_position.tolist()
            if final_position is not None
            else None
        ),
    "placement_error_xy_m":
        placement_error,
    "success_threshold_m":
        SUCCESS_THRESHOLD,
    "success":
        bool(success),
    "failure_type":
        None if success else "relative_placement_failed",
    "simulation_steps":
        step_count,
    "duration_sec":
        duration_sec,
    "environment":
        workcell_metadata,
    "autonomy_features": {
        "language_parsing": True,
        "scene_inspection": True,
        "dynamic_target_calculation": True,
        "skill_selection": True,
        "physical_execution": True,
        "post_action_verification": True,
        "failure_recovery": False,
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

print("\n[Agent] Final result:")
print(json.dumps(result, indent=2))

print(f"\n[Agent] Result saved: {result_path}")

simulation_app.close()
