"""
@FileName：allegro_hand.py
@Description：
@Author：Ferry
@Time：4/18/25 2:07 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from Env.allegro_hand_cfg import ALLEGRO_HAND_CFG

def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()






def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=0.5)

    # Origin 1 with Allegro Hand
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # -- Robot
    allegro_hand = Articulation(ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot"))



    # return the scene information
    scene_entities = {
        # "allegro": allegro,
        "allegro_hand": allegro_hand,
        # "contact_sensor": con,
        # 'terrain': terrain,
    }
    return scene_entities, origins






def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str,], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.1, .1, .1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Simulate physics
    while simulation_app.is_running():
        # reset
        robot = entities['allegro_hand']

        ee_marker.visualize(robot.data.body_pos_w[0, 0, :].reshape(-1, 3) + origins[0],
                            robot.data.body_quat_w[0, 0, :].reshape(-1, 4))
        # goal_marker.visualize(pid_controller.goal[:, :3] + origins[0],
        #                       pid_controller.goal[:, 3:7])
        robot.write_root_pose_to_sim(torch.tensor((0.0, 0.0, 0.0,  0,  0,  0, 1.0)).reshape(-1,7))
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in [entities['allegro_hand']]:
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


# PID K params
K_p_pos, K_i_pos, K_d_pos = 10.0, 0.0001, 7.0  # pos p i d of  PID
K_p_rot, K_i_rot, K_d_rot = 0.5, 0.025, 0.65  # rot p i d of  PID
dt = 0.01
threshold = 0.01
threshold_ang = 5

if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
    print(123)
