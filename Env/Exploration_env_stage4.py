"""
@FileName：Exploration_env_stage4.py.py
@Description：
@Author：Ferry
@Time：2025 5/2/25 11:16 AM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import argparse

from isaaclab.app import AppLauncher
from utils.Controller import Pid_Controller
from utils.utils import *

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from allegro_hand_cfg import ALLEGRO_HAND_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from collections import deque
from sklearn.neighbors import KDTree


@configclass
class ExpllorationEnvCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True,
                                                         max_linear_velocity=.0, max_angular_velocity=.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, 1.0, .0)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    index_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/index_link_3",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_air_time=True,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]
    )
    middle_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/middle_link_3",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_air_time=True,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]
    )
    ring_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ring_link_3",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_air_time=True,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]
    )
    thumb_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/thumb_link_3",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
        track_air_time=True,
        track_pose=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/object"]
    )


"""
--------------
# move to func
--------------
"""


def move_to(robot, controller, pos, quat):
    controller.reset(torch.cat((pos, quat), dim=-1).reshape(-1, 7), [0])
    force, torques = controller.step(robot.data.root_state_w)
    # force, torques = transform_force_torque_to_local_gpu(force, torques, robot.data.root_quat_w)
    robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
                                        torques=torques.reshape(-1, 1, 3).to("cuda:0"),
                                        body_ids=[0])


"""
-------------------------------
# Generate next exploration point
-------------------------------
"""


def calculate_points_score(
        origin_points, surface_points, uncertainty, prediction, hand_pos_now, explored_queue,
        alpha=.5, beta=1.0, gamma=.5, delta=2.0, epsilon=0.05,
        proximity_radius=0.05
):
    # Uncertainty score U(x)
    U = uncertainty
    # Distance penalty D(x)
    D = -np.linalg.norm(origin_points - hand_pos_now.squeeze(), axis=1)
    # Information score I(x)
    I = np.exp(-np.abs(prediction) / epsilon) * uncertainty * 1e10

    U_norm = normalize(U)
    D_norm = normalize(D)
    I_norm = normalize(I)

    S = alpha * U_norm + beta * D_norm + gamma * I_norm

    # closest explored point per surface_point
    tree_explored = KDTree(explored_queue)
    distances, _ = tree_explored.query(origin_points, k=1)
    P = np.where(distances.squeeze() < proximity_radius, -1.0, 0.0)
    S = S + delta * P

    return S


def sample_next_exploration_point(origin, surface_points, uncertainty, prediction, estimated_surface_uncertainty,
                                  etimated_surface_prediction, hand_pos_now, explored_queue, hand_quat):
    explored_queue_np = np.array(explored_queue).reshape(-1, 3)
    selected_idx = kd_pruning(surface_points)
    surface_points_pruned, uncertainty_pruned, prediction_pruned = surface_points[selected_idx], \
    estimated_surface_uncertainty[selected_idx], etimated_surface_prediction[selected_idx]
    origin_cutdown, uncertainty_cutdown, prediction_cutdown = kd_cut_down(
        origin,
        explored_queue_np,
        uncertainty,
        prediction,
        k=100
    )

    S = calculate_points_score(origin_cutdown, surface_points_pruned, uncertainty_cutdown, prediction_cutdown,
                               hand_pos_now,
                               explored_queue_np)

    best_idx = np.argmax(S)
    pos_target = origin[best_idx]

    quat_target = calculate_rot(pos_target, surface_points_pruned, hand_quat)

    return (torch.tensor(pos_target, dtype=torch.float32, device='cuda').reshape(-1, 3),
            torch.tensor(quat_target, dtype=torch.float32, device='cuda').reshape(-1, 4))


def run_simulator(sim: sim_utils.SimulationContext, entities):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Define the frame marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.1, .1, .1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Init Robot state
    robot = entities['robot']
    alpha = 0.05
    pos = torch.tensor([0, .2, -0.48], device="cuda")
    quat = torch.tensor((0.257551, 0.283045, 0.683330, -0.621782), device="cuda")
    grasp_joint_pos = robot.data.default_joint_pos.clone()
    target_pos, target_quat = pos.reshape(-1, 3) + entities.env_origins[0], quat.reshape(-1, 4)

    # Initialize buffer
    touched_buf = torch.empty((0, 3), device="cuda")
    untouched_buf = torch.empty((0, 3), device="cuda")
    explored_queue = deque(maxlen=10)

    # init global_HE_GPIS model
    temp_min = np.array([-0.52, -0.52, -0.52])
    temp_max = np.array([0.52, 0.52, 0.52])
    gpis = init_global_HE_GPIS_model(temp_min, temp_max, store_path="../Results/Exploration_env_stage4_")

    # init Hand position Controller (local control in isaaclab)
    K_p_pos, K_i_pos, K_d_pos = 10.0, 0.0001, 7.0  # pos p i d of  PID
    K_p_rot, K_i_rot, K_d_rot = 0.5, 0.025, 0.65  # rot p i d of  PID
    dt = 0.01
    controller = Pid_Controller(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, local_control=True)

    # Simulate physics
    while simulation_app.is_running():

        if get_exploration_status(count):  # time to next exploration
            # update explored queue
            explored_queue.append(robot.data.body_pos_w[0, 0, :].reshape(-1, 3).to("cpu"))

            # update buffer
            sim_time = 0.0
            count = 0
            update_buffer_x = np.vstack([touched_buf.cpu().numpy(), untouched_buf.cpu().numpy()])
            update_buffer_y = np.hstack(
                [np.zeros((touched_buf.cpu().numpy().shape[0])), np.ones((untouched_buf.cpu().numpy().shape[0]))])

            # update gpis model and generate estimated_surface
            uncertainty, xstar, estimated_surface, estimated_surface_uncertainty, prediction, estimated_surface_prediction = gpis.step(
                update_buffer_x, update_buffer_y)

            # generation next exploration position
            if len(estimated_surface) > 0:
                target_pos, target_quat = sample_next_exploration_point(
                    xstar,
                    estimated_surface,
                    uncertainty,
                    prediction,
                    estimated_surface_uncertainty,
                    estimated_surface_prediction,
                    robot.data.body_pos_w[0, 0, :].reshape(-1, 3).to("cpu").numpy(),
                    explored_queue,
                    robot.data.body_quat_w[0, 0, :].reshape(-1, 4).to("cpu").numpy()
                )
                target_pos = target_pos + entities.env_origins[0]
                target_quat_check = check_quat_validity(target_quat)
                if target_quat_check is not None:
                    target_quat = target_quat_check

            grasp_joint_pos = robot.data.default_joint_pos.clone()

            touched_buf = torch.empty((0, 3), device="cuda")
            untouched_buf = torch.empty((0, 3), device="cuda")


        elif get_grasp_status(count):
            # generate joint positions
            grasp_joint_pos = robot.data.soft_joint_pos_limits[..., 1] - 0.9

        pre_joint_pos_target = (1 - alpha) * robot.data.joint_pos + alpha * grasp_joint_pos
        robot.set_joint_position_target(pre_joint_pos_target)

        move_to(robot, controller, target_pos, target_quat)

        if count % 10 == 0:
            touched_buf, untouched_buf = store_contact_points(robot, entities, touched_buf, untouched_buf)

        # write data to sim
        count, sim_time = env_step(ee_marker, goal_marker, robot, robot.data.body_pos_w, robot.data.body_quat_w,
                                   entities, target_pos, target_quat, sim, sim_time,
                                   sim_dt, count, scene)


if __name__ == '__main__':
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_cfg = ExpllorationEnvCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Run the simulator
    run_simulator(sim, scene)

    simulation_app.close()
