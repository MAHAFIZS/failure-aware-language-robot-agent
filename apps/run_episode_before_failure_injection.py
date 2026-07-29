# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one language-conditioned Franka pick-and-place episode."
    )
    parser.add_argument(
        "--command",
        type=str,
        default="Pick up the cube and place it at the target",
        help="Natural-language task command.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the Isaac Sim GUI.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Maximum simulation steps before timeout.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=120,
        help="Physics steps to wait after controller completion.",
    )
    parser.add_argument(
        "--success-threshold-m",
        type=float,
        default=0.06,
        help="Maximum horizontal cube-to-target error for success.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/latest_episode.json"),
        help="Episode report path.",
    )

    args, _ = parser.parse_known_args()
    return args


args = parse_arguments()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
    }
)

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import (
    PickPlaceController,
)
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path


SUPPORTED_COMMANDS = {
    "pick up the cube and place it at the target",
    "pick up the blue cube and place it at the target",
    "move the cube to the target",
}


def normalize_command(command: str) -> str:
    return " ".join(command.lower().strip().rstrip(".").split())


def validate_command(command: str) -> dict:
    normalized = normalize_command(command)

    if normalized not in SUPPORTED_COMMANDS:
        raise ValueError(
            "Unsupported command. Current prototype supports only "
            "cube-to-target pick-and-place commands."
        )

    return {
        "task_type": "pick_place",
        "object_id": "cube",
        "target_id": "target",
        "normalized_command": normalized,
    }


def horizontal_error(position: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(position[:2] - target[:2]))


def to_json_safe(value):
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            report,
            indent=2,
            default=to_json_safe,
        ),
        encoding="utf-8",
    )


def main() -> int:
    command_task = validate_command(args.command)

    assets_root_path = get_assets_root_path()

    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return 2

    cube_size_m = 0.0515

    target_position = np.array(
        [-0.3, -0.3, cube_size_m / 2.0],
        dtype=float,
    )

    initial_cube_position = np.array(
        [0.3, 0.3, 0.3],
        dtype=float,
    )

    world = World(
        stage_units_in_meters=1.0,
    )

    world.scene.add_default_ground_plane()

    asset_path = (
        assets_root_path
        + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    )

    robot_prim = add_reference_to_stage(
        usd_path=asset_path,
        prim_path="/World/Franka",
    )

    robot_prim.GetVariantSet("Gripper").SetVariantSelection(
        "AlternateFinger"
    )
    robot_prim.GetVariantSet("Mesh").SetVariantSelection(
        "Quality"
    )

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        joint_prim_names=[
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        joint_opened_positions=np.array(
            [0.05, 0.05],
            dtype=float,
        ),
        joint_closed_positions=np.array(
            [0.02, 0.02],
            dtype=float,
        ),
        action_deltas=np.array(
            [0.01, 0.01],
            dtype=float,
        ),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path="/World/Franka",
            name="my_franka",
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            gripper=gripper,
        )
    )

    cube = world.scene.add(
        DynamicCuboid(
            name="cube",
            position=initial_cube_position,
            prim_path="/World/Cube",
            scale=np.array(
                [cube_size_m, cube_size_m, cube_size_m],
                dtype=float,
            ),
            size=1.0,
            color=np.array(
                [0.0, 0.0, 1.0],
                dtype=float,
            ),
        )
    )

    franka.gripper.set_default_state(
        franka.gripper.joint_opened_positions
    )

    world.reset()

    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
    )

    articulation_controller = (
        franka.get_articulation_controller()
    )

    controller_completed = False
    completion_step = None
    step_count = 0
    status = "running"

    print(f"Command: {args.command}")
    print(f"Task: {command_task}")
    print(f"Target position: {target_position.tolist()}")

    while (
        simulation_app.is_running()
        and step_count < args.max_steps
    ):
        world.step(render=not args.headless)
        step_count += 1

        if not world.is_playing():
            continue

        if not controller_completed:
            action = controller.forward(
                picking_position=np.asarray(
                    cube.get_local_pose()[0],
                    dtype=float,
                ),
                placing_position=target_position,
                current_joint_positions=franka.get_joint_positions(),
                end_effector_offset=np.array(
                    [0.0, 0.005, 0.0],
                    dtype=float,
                ),
            )

            articulation_controller.apply_action(action)

            if controller.is_done():
                controller_completed = True
                completion_step = step_count

                print(
                    "Controller completed. "
                    "Waiting for the cube to settle."
                )

        else:
            if completion_step is None:
                raise RuntimeError(
                    "Controller completed without a completion step."
                )

            settled_steps = step_count - completion_step

            if settled_steps >= args.settle_steps:
                status = "completed"
                break

    if status != "completed":
        status = "timeout"

    final_cube_position = np.asarray(
        cube.get_world_pose()[0],
        dtype=float,
    )

    xy_error_m = horizontal_error(
        final_cube_position,
        target_position,
    )

    vertical_error_m = float(
        abs(
            float(final_cube_position[2])
            - float(target_position[2])
        )
    )

    placement_success = bool(
        controller_completed
        and status == "completed"
        and xy_error_m <= float(args.success_threshold_m)
        and float(final_cube_position[2]) < 0.10
    )

    if placement_success:
        failure_type = None
    elif status == "timeout":
        failure_type = "timeout"
    else:
        failure_type = "placement_error"

    report = {
        "schema_version": "0.1",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "command": args.command,
        "task": command_task,
        "simulator": {
            "name": "NVIDIA Isaac Sim",
            "stage_units_in_meters": 1.0,
            "headless": bool(args.headless),
        },
        "episode": {
            "status": status,
            "steps": int(step_count),
            "controller_completed": bool(
                controller_completed
            ),
            "attempts": 1,
        },
        "initial_state": {
            "cube_position_m": (
                initial_cube_position.tolist()
            ),
            "target_position_m": (
                target_position.tolist()
            ),
        },
        "final_state": {
            "cube_position_m": (
                final_cube_position.tolist()
            ),
        },
        "metrics": {
            "horizontal_position_error_m": float(
                xy_error_m
            ),
            "vertical_position_error_m": float(
                vertical_error_m
            ),
            "success_threshold_m": float(
                args.success_threshold_m
            ),
        },
        "outcome": {
            "placement_success": bool(
                placement_success
            ),
            "failure_type": failure_type,
        },
    }

    write_report(
        args.output,
        report,
    )

    print(json.dumps(report, indent=2))
    print(f"Saved episode report to: {args.output}")

    return 0 if placement_success else 1


if __name__ == "__main__":
    exit_code = 3

    try:
        exit_code = main()

    except ValueError as exc:
        print(
            f"Command error: {exc}",
            file=sys.stderr,
        )
        exit_code = 2

    except Exception as exc:
        carb.log_error(
            f"Episode failed with unexpected error: {exc}"
        )
        exit_code = 3

    finally:
        simulation_app.close()

    raise SystemExit(exit_code)
