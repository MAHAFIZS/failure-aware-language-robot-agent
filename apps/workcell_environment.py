from __future__ import annotations

import numpy as np
import omni.usd

from pxr import Gf, Sdf, UsdLux

from isaacsim.core.api.objects import (
    FixedCuboid,
    VisualCuboid,
)
from isaacsim.core.utils.viewports import (
    set_camera_view,
)


def add_fixed_box(
    world,
    *,
    name: str,
    prim_path: str,
    position: list[float],
    scale: list[float],
    color: list[float],
) -> FixedCuboid:
    return world.scene.add(
        FixedCuboid(
            name=name,
            prim_path=prim_path,
            position=np.asarray(
                position,
                dtype=float,
            ),
            scale=np.asarray(
                scale,
                dtype=float,
            ),
            size=1.0,
            color=np.asarray(
                color,
                dtype=float,
            ),
        )
    )


def add_visual_box(
    world,
    *,
    name: str,
    prim_path: str,
    position: list[float],
    scale: list[float],
    color: list[float],
) -> VisualCuboid:
    return world.scene.add(
        VisualCuboid(
            name=name,
            prim_path=prim_path,
            position=np.asarray(
                position,
                dtype=float,
            ),
            scale=np.asarray(
                scale,
                dtype=float,
            ),
            size=1.0,
            color=np.asarray(
                color,
                dtype=float,
            ),
        )
    )


def add_workcell_lighting() -> None:
    stage = omni.usd.get_context().get_stage()

    key_light = UsdLux.DistantLight.Define(
        stage,
        Sdf.Path("/World/Lights/KeyLight"),
    )
    key_light.CreateIntensityAttr(850.0)
    key_light.CreateAngleAttr(1.5)
    key_light.CreateColorAttr(
        Gf.Vec3f(1.0, 0.95, 0.88)
    )

    fill_light = UsdLux.RectLight.Define(
        stage,
        Sdf.Path("/World/Lights/FillLight"),
    )
    fill_light.CreateIntensityAttr(1400.0)
    fill_light.CreateWidthAttr(2.5)
    fill_light.CreateHeightAttr(1.5)
    fill_light.CreateColorAttr(
        Gf.Vec3f(0.80, 0.88, 1.0)
    )

    fill_prim = fill_light.GetPrim()
    xform = fill_prim.GetAttribute(
        "xformOp:translate"
    )

    if not xform:
        from pxr import UsdGeom

        xformable = UsdGeom.Xformable(fill_prim)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(
            Gf.Vec3d(0.4, 0.0, 2.4)
        )
    else:
        xform.Set(
            Gf.Vec3d(0.4, 0.0, 2.4)
        )


def build_workcell(
    world,
    *,
    add_sorting_trays: bool = True,
) -> dict:
    """
    Build a reusable industrial-style Franka workcell.

    The task objects remain at approximately z=0, so the
    tabletop top surface is also kept at z=0. This avoids
    changing the already verified manipulation trajectories.
    """

    # -----------------------------------------------------------------
    # Safety floor
    # -----------------------------------------------------------------

    add_visual_box(
        world,
        name="safety_floor",
        prim_path="/World/Workcell/SafetyFloor",
        position=[0.15, 0.0, -0.055],
        scale=[2.4, 2.0, 0.05],
        color=[0.16, 0.17, 0.18],
    )

    # Yellow marked robot safety zone.
    add_visual_box(
        world,
        name="safety_zone",
        prim_path="/World/Workcell/SafetyZone",
        position=[0.15, 0.0, -0.027],
        scale=[1.75, 1.45, 0.008],
        color=[0.90, 0.62, 0.03],
    )

    # Dark central work zone over the yellow boundary.
    add_visual_box(
        world,
        name="work_zone",
        prim_path="/World/Workcell/WorkZone",
        position=[0.15, 0.0, -0.021],
        scale=[1.55, 1.25, 0.008],
        color=[0.22, 0.24, 0.26],
    )

    # -----------------------------------------------------------------
    # Work table
    # -----------------------------------------------------------------

    # Top surface ends at z=0, preserving current cube positions.
    add_fixed_box(
        world,
        name="work_table_top",
        prim_path="/World/Workcell/Table/Top",
        position=[0.32, 0.0, -0.035],
        scale=[1.35, 1.05, 0.07],
        color=[0.34, 0.37, 0.40],
    )

    # Visible front apron makes the table look substantial.
    add_visual_box(
        world,
        name="table_front_apron",
        prim_path="/World/Workcell/Table/FrontApron",
        position=[0.32, -0.515, -0.14],
        scale=[1.35, 0.04, 0.25],
        color=[0.12, 0.13, 0.14],
    )

    # Table legs remain mostly below the task surface.
    leg_positions = [
        [-0.27, -0.43, -0.37],
        [-0.27, 0.43, -0.37],
        [0.91, -0.43, -0.37],
        [0.91, 0.43, -0.37],
    ]

    for index, position in enumerate(
        leg_positions
    ):
        add_visual_box(
            world,
            name=f"table_leg_{index}",
            prim_path=(
                f"/World/Workcell/Table/Leg{index}"
            ),
            position=position,
            scale=[0.07, 0.07, 0.68],
            color=[0.10, 0.11, 0.12],
        )

    # -----------------------------------------------------------------
    # Walls
    # -----------------------------------------------------------------

    add_visual_box(
        world,
        name="back_wall",
        prim_path="/World/Workcell/Walls/Back",
        position=[0.2, 0.78, 0.75],
        scale=[2.2, 0.05, 1.5],
        color=[0.68, 0.70, 0.72],
    )

    add_visual_box(
        world,
        name="left_wall",
        prim_path="/World/Workcell/Walls/Left",
        position=[-0.83, 0.0, 0.75],
        scale=[0.05, 1.55, 1.5],
        color=[0.58, 0.61, 0.64],
    )

    add_visual_box(
        world,
        name="right_wall",
        prim_path="/World/Workcell/Walls/Right",
        position=[1.23, 0.0, 0.75],
        scale=[0.05, 1.55, 1.5],
        color=[0.58, 0.61, 0.64],
    )

    # -----------------------------------------------------------------
    # Safety frame
    # -----------------------------------------------------------------

    frame_color = [0.93, 0.68, 0.05]

    post_positions = [
        [-0.72, -0.68, 0.75],
        [-0.72, 0.68, 0.75],
        [1.12, -0.68, 0.75],
        [1.12, 0.68, 0.75],
    ]

    for index, position in enumerate(
        post_positions
    ):
        add_visual_box(
            world,
            name=f"safety_post_{index}",
            prim_path=(
                f"/World/Workcell/SafetyFrame/"
                f"Post{index}"
            ),
            position=position,
            scale=[0.045, 0.045, 1.5],
            color=frame_color,
        )

    add_visual_box(
        world,
        name="safety_frame_front",
        prim_path=(
            "/World/Workcell/SafetyFrame/"
            "TopFront"
        ),
        position=[0.20, -0.68, 1.48],
        scale=[1.88, 0.045, 0.045],
        color=frame_color,
    )

    add_visual_box(
        world,
        name="safety_frame_back",
        prim_path=(
            "/World/Workcell/SafetyFrame/"
            "TopBack"
        ),
        position=[0.20, 0.68, 1.48],
        scale=[1.88, 0.045, 0.045],
        color=frame_color,
    )

    # -----------------------------------------------------------------
    # Sorting trays
    # -----------------------------------------------------------------

    tray_data = []

    if add_sorting_trays:
        trays = [
            (
                "red",
                [-0.25, 0.28, 0.008],
                [0.55, 0.08, 0.08],
            ),
            (
                "green",
                [-0.25, 0.00, 0.008],
                [0.08, 0.45, 0.12],
            ),
            (
                "blue",
                [-0.25, -0.28, 0.008],
                [0.08, 0.18, 0.60],
            ),
        ]

        for label, position, color in trays:
            tray = add_visual_box(
                world,
                name=f"{label}_workcell_tray",
                prim_path=(
                    f"/World/Workcell/Trays/"
                    f"{label.capitalize()}Tray"
                ),
                position=position,
                scale=[0.17, 0.17, 0.016],
                color=color,
            )

            tray_data.append(
                {
                    "label": label,
                    "prim_path": tray.prim_path,
                    "position": position,
                }
            )

    # -----------------------------------------------------------------
    # Lighting and viewport
    # -----------------------------------------------------------------

    add_workcell_lighting()

    set_camera_view(
        eye=np.array(
            [1.55, -1.60, 1.25],
            dtype=float,
        ),
        target=np.array(
            [0.30, 0.00, 0.18],
            dtype=float,
        ),
        camera_prim_path="/OmniverseKit_Persp",
    )

    return {
        "environment_name":
            "industrial_franka_workcell_v0.1",
        "table":
            True,
        "safety_floor":
            True,
        "surrounding_walls":
            True,
        "safety_frame":
            True,
        "sorting_trays":
            bool(add_sorting_trays),
        "camera_view":
            {
                "eye": [1.55, -1.60, 1.25],
                "target": [0.30, 0.00, 0.18],
            },
        "lighting":
            {
                "distant_light_intensity":
                    850.0,
                "rect_light_intensity":
                    1400.0,
            },
        "materials":
            "color-based industrial material approximation",
        "trays":
            tray_data,
    }
