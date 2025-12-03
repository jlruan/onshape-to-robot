from __future__ import annotations


import time
from pathlib import Path
import eaik
import numpy as np
import viser
import tyro
from viser.extras import ViserUrdf


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []

    initial_config: list[float] = []

    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():

        lower = lower if lower is not None else -np.pi

        upper = upper if upper is not None else np.pi

        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )

        def tmp(_):
            config = np.array([slider.value for slider in slider_handles])
            fk = bot.fwdKin(config)
            if fk is not None:
                server.scene.add_point_cloud(
                    "fk",
                    points=fk[:3, 3].reshape(1, 3),
                    colors=(0, 255, 0),
                    point_size=0.025,
                )
            viser_urdf.update_cfg(config)

        slider.on_update(tmp)
        slider_handles.append(slider)

        initial_config.append(initial_pos)

    return slider_handles, initial_config


def main(
    urdf_path: Path = Path("urdf/r2000ic125l/spherized.urdf"),
    load_meshes: bool = True,
    load_collision_meshes: bool = True,
) -> None:
    """Visualize a URDF in viser.

    Args:
        urdf_path: Path to the URDF file.
        load_meshes: Whether to load visual meshes.
        load_collision_meshes: Whether to load collision meshes.
    """

    # Start viser server.
    global server
    server = viser.ViserServer()
    print(f"Viser server started (check terminal for URL)")

    # Load URDF
    print(f"Loading URDF from: {urdf_path}")
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    global bot
    # H = np.array(
    #     [
    #         [0, 0, 0, -1, 0, -1],
    #         [0, 1, 1, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0],
    #     ]
    # )

    # P = np.array(
    #     [
    #         [0.000, 0.312, 0.000, 1.499, 0.231, 0.215, 0.000],
    #         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    #         [0.299, 0.371, 1.075, 0.225, 0.000, 0.000, 0.000],
    #     ]
    # )
    # bot = eaik.HPRobot(H, P, None)
    bot = eaik.UrdfRobot(urdf_path)
    print("Is spherical:", bot.hasSphericalWrist())
    print("Kinematic family:", bot.getKinematicFamily())
    print("✓ URDF loaded successfully")
    # Create sliders in GUI that help us move the robot joints.

    with server.gui.add_folder("Joint position control"):

        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.

    with server.gui.add_folder("Visibility"):

        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )

        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.

    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create coordinate frame at origin.
    server.scene.add_frame(
        "/visual/link_1/link_2/link_3/link_4/link_5/link_6/flange",
        axes_length=0.25,
        axes_radius=0.01,
    )

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene

    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.

    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):

        for s, init_q in zip(slider_handles, initial_config):

            s.value = init_q

    # Sleep forever.

    while True:
        time.sleep(10)


if __name__ == "__main__":
    tyro.cli(main)
