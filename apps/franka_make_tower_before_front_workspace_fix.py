# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# Parse our arguments before starting Isaac Sim.
parser = argparse.ArgumentParser(
    description="Stack colored cubes into a tower."
)

parser.add_argument(
    "--instruction",
    default="Make a tower with all colored boxes.",
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
    default="results/latest_isaac_sort_result.json",
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=30000,
)

args, unknown = parser.parse_known_args()


from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": bool(args.headless),
    }
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


np.random.seed(args.seed)

CUBE_SIZE = 0.0515
CUBE_Z = CUBE_SIZE / 2.0
SPAWN_Z = 0.30

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


assets_root_path = get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder.")
    simulation_app.close()
    raise SystemExit(1)


# ---------------------------------------------------------------------
# World and Franka
# ---------------------------------------------------------------------

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

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
    joint_opened_positions=np.array(
        [0.05, 0.05]
    ),
    joint_closed_positions=np.array(
        [0.02, 0.02]
    ),
    action_deltas=np.array(
        [0.01, 0.01]
    ),
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
# Boxes and matching destination bins
# ---------------------------------------------------------------------

sorting_items = [
    {
        "label": "red",
        "color": np.array([1.0, 0.0, 0.0]),
        "start": np.array([0.35, 0.28, SPAWN_Z]),
        "target": np.array([-0.25, 0.00, CUBE_Z]),
    },
    {
        "label": "green",
        "color": np.array([0.0, 1.0, 0.0]),
        "start": np.array([0.45, 0.00, SPAWN_Z]),
        "target": np.array(
            [-0.25, 0.00, CUBE_Z + CUBE_SIZE]
        ),
    },
    {
        "label": "blue",
        "color": np.array([0.0, 0.2, 1.0]),
        "start": np.array([0.35, -0.25, SPAWN_Z]),
        "target": np.array(
            [-0.25, 0.00, CUBE_Z + 2.0 * CUBE_SIZE]
        ),
    },
]

cubes: list[DynamicCuboid] = []

for item in sorting_items:
    label = item["label"]

    cube = world.scene.add(
        DynamicCuboid(
            name=f"{label}_cube",
            prim_path=f"/World/{label.capitalize()}Cube",
            position=item["start"],
            scale=np.array(
                [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]
            ),
            size=1.0,
            color=item["color"],
        )
    )

    cubes.append(cube)




franka.gripper.set_default_state(
    franka.gripper.joint_opened_positions
)

world.reset()


controller = PickPlaceController(
    name="multi_box_sort_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
)

articulation_controller = (
    franka.get_articulation_controller()
)


# ---------------------------------------------------------------------
# Sequential sorting loop
# ---------------------------------------------------------------------

current_item_index = 0
completed_items: list[dict] = []

step_count = 0
start_time = time.time()
settle_steps_remaining = 0
initial_settle_steps = 120
finished = False

print("\n[Isaac Tower] Instruction:")
print(args.instruction)

print("\n[Isaac Tower] Planned sequence:")

for index, item in enumerate(sorting_items, start=1):
    print(
        f"  {index}. Move {item['label']} cube "
        f"to {item['label']} bin"
    )


while simulation_app.is_running():
    world.step(render=not args.headless)
    step_count += 1

    if step_count >= args.max_steps:
        print(
            "[Isaac Tower] Maximum step count reached."
        )
        break

    if not world.is_playing():
        continue

    if initial_settle_steps > 0:
        initial_settle_steps -= 1
        continue

    if finished:
        # Allow a short visible pause after completion.
        settle_steps_remaining -= 1

        if settle_steps_remaining <= 0:
            break

        continue

    item = sorting_items[current_item_index]
    cube = cubes[current_item_index]

    picking_position = np.asarray(
        cube.get_local_pose()[0],
        dtype=float,
    )

    placing_position = np.asarray(
        item["target"],
        dtype=float,
    )

    actions = controller.forward(
        picking_position=picking_position,
        placing_position=placing_position,
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
            cube.get_local_pose()[0],
            dtype=float,
        )

        placement_error_xy = distance_xy(
            final_position,
            placing_position,
        )

        item_success = placement_error_xy <= 0.10

        completed_items.append(
            {
                "label": item["label"],
                "cube_name": f"{item['label']}_cube",
                "bin_name": f"{item['label']}_bin",
                "target_position":
                    placing_position.tolist(),
                "final_position":
                    final_position.tolist(),
                "placement_error_xy_m":
                    placement_error_xy,
                "success":
                    item_success,
            }
        )

        print(
            f"[Isaac Tower] {item['label']} cube finished: "
            f"error={placement_error_xy:.4f} m, "
            f"success={item_success}"
        )

        current_item_index += 1

        if current_item_index >= len(
            sorting_items
        ):
            finished = True
            settle_steps_remaining = 120

            print(
                "[Isaac Tower] Tower sequence completed."
            )
        else:
            controller.reset()

            # Ensure the next task starts with an open gripper.
            franka.gripper.open()

            print(
                "[Isaac Tower] Next object: "
                f"{sorting_items[current_item_index]['label']}"
            )


duration_sec = time.time() - start_time

success_count = sum(
    bool(item["success"])
    for item in completed_items
)

all_sorted = (
    len(completed_items) == len(sorting_items)
    and success_count == len(sorting_items)
)

result = {
    "schema_version":
        "contacttrace_isaac_tower_episode_v0.1",
    "simulator":
        "isaac_sim",
    "robot":
        "franka_panda",
    "instruction":
        args.instruction,
    "seed":
        int(args.seed),
    "task_type":
        "multi_object_tower_stack",
    "planner":
        "rule_based_tower_sequence",
    "requested_object_count":
        len(sorting_items),
    "completed_object_count":
        len(completed_items),
    "successful_object_count":
        success_count,
    "success":
        all_sorted,
    "failure_type":
        None if all_sorted else "tower_incomplete",
    "duration_sec":
        duration_sec,
    "simulation_steps":
        step_count,
    "items":
        completed_items,
}

result_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_path.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n[Isaac Tower] Final ContactTrace result:")
print(
    json.dumps(
        result,
        indent=2,
    )
)

print(
    f"\n[Isaac Tower] Result saved: {result_path}"
)

simulation_app.close()
