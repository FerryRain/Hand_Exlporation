"""
@FileName：Exploration_env_stage8.py.py
@Description：
@Author：Ferry
@Time：2025 5/20/25 4:52 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import argparse

from isaaclab.app import AppLauncher
from utils.Controller import HybridForcePositionController
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
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from collections import deque


@configclass
class ExpllorationEnvCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = RigidObjectCollectionCfg(
        rigid_objects={"dot": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.SphereCfg(
                radius=0.05,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, .0, 1.0)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.16)),
        )
        }
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True,
                                                         max_linear_velocity=.0, max_angular_velocity=.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, 1.0, .0)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
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


def base_move(robot, controller):
    force, torques = controller.step_super(robot.data.object_state_w.reshape(-1, 13))
    robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
                                        torques=torques.reshape(-1, 1, 3).to("cuda:0"),
                                        object_ids=[0])


def move_to_base(robot, controller, contact_force, hand_pos_now, gpis, desired_penetration=0.005, f_target=2.5):
    if contact_detect(contact_force):
        f_target = f_target / 1
    else:
        f_target = f_target * 2
    prediction, uncertainty, variance_gradients, miu_gradients, miu_normals = gpis.predict_points(
        hand_pos_now.reshape(-1, 3))
    d_Normal = -(miu_normals / (np.linalg.norm(miu_normals) + 1e-10))
    robot_state = robot.data.object_state_w.reshape(-1, 13)
    contact_force_tensor = contact_force.clone().detach().reshape(-1, 3)
    surface_normal = torch.tensor(d_Normal, dtype=torch.float32, device='cuda').reshape(-1, 3)

    force, torques = controller.step_with_contact(
        robot_state,
        contact_force_tensor,
        surface_normal,
        desired_penetration=desired_penetration,
        f_target=f_target
    )
    robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
                                        torques=torques.reshape(-1, 1, 3).to("cuda:0"),
                                        object_ids=[0])


def move_to_new(robot, controller, quat, gpis, hand_pos_now, entities, contact_force, beta_base=0.08,
                desired_penetration=0.005, f_target=2.5):
    if contact_detect(contact_force):
        f_target = f_target / 1
        beta = beta_base
    else:
        f_target = f_target * 2
        beta = beta_base
    prediction, uncertainty, variance_gradients, miu_gradients, miu_normals = gpis.predict_points(
        hand_pos_now.reshape(-1, 3))
    d_Normal = -(miu_normals / (np.linalg.norm(miu_normals) + 1e-10))
    d_Tagent = compute_projected_unit_direction_np(variance_gradients, miu_normals)
    delty_y_Tagent = beta * d_Tagent

    pos_target_PID = hand_pos_now + delty_y_Tagent

    pos_target_PID = torch.tensor(pos_target_PID, dtype=torch.float32, device='cuda').reshape(-1, 3)
    target_pos_PID = pos_target_PID + entities.env_origins[0]
    target_quat = check_quat_validity(quat)
    controller.reset(torch.cat((target_pos_PID, target_quat), dim=-1).reshape(-1, 7), [0])

    robot_state = robot.data.object_state_w.reshape(-1, 13)
    contact_force_tensor = contact_force.clone().detach().reshape(-1, 3)
    surface_normal = torch.tensor(d_Normal, dtype=torch.float32, device='cuda').reshape(-1, 3)

    force, torques = controller.step_with_contact(
        robot_state,
        contact_force_tensor,
        surface_normal,
        desired_penetration=desired_penetration,
        f_target=f_target
    )
    robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
                                        torques=torques.reshape(-1, 1, 3).to("cuda:0"),
                                        object_ids=[0])
    return target_pos_PID


def run_simulator(sim: sim_utils.SimulationContext, entities):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Define the frame marker
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.01, .01, .01)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Init Robot state
    robot = entities['robot']
    pos = torch.tensor([0, 0.15, 0.0], device="cuda")
    quat = torch.tensor((0.257551, 0.283045, 0.683330, -0.621782), device="cuda")
    target_pos, target_quat = pos.reshape(-1, 3) + entities.env_origins[0], quat.reshape(-1, 4)

    # Initialize buffer
    touched_buf = torch.empty((0, 3), device="cuda")
    untouched_buf = torch.empty((0, 3), device="cuda")
    explored_queue = deque(maxlen=10)

    # init global_HE_GPIS model
    temp_min = np.array([-0.2, -0.2, -0.2])
    temp_max = np.array([0.2, 0.2, 0.2])
    temp_bb_min = np.array([-0.15, -0.15, -0.15])
    temp_bb_max = np.array([0.15, 0.15, 0.15])

    gpis = init_normal_HE_GPIS_model_2(temp_min, temp_max, temp_bb_min, temp_bb_max, res=100, grid_count=6,
                                       store_path="../Results/Exploration_env_stage8_")

    # init Hand position Controller (local control in isaaclab)
    # K_p_pos, K_i_pos, K_d_pos = 10.0, 0.0001, 7.0  # pos p i d of  PID
    # K_p_rot, K_i_rot, K_d_rot = 0.5, 0.025, 0.65  # rot p i d of  PID
    K_p_pos, K_i_pos, K_d_pos = 1.25, 0.00001, 1.4  # pos p i d of  PID
    K_p_rot, K_i_rot, K_d_rot = 0.1, 0.0, 0.1  # rot p i d of  PID
    k_f, K_imp, D_imp= 1.0, 4.0, 2.0
    # k_f, K_imp, D_imp = 0, 0, 0
    dt = 0.01
    controller = HybridForcePositionController(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, k_f=k_f,
                                               K_imp=K_imp, D_imp=D_imp, local_control=True)
    controller.reset(torch.cat((target_pos, target_quat), dim=-1).reshape(-1, 7), [0])
    Hybrid_control = False

    estimated_surface = []
    # Simulate physics
    while simulation_app.is_running():

        if get_exploration_status(count):  # time to next exploration
            explored_queue.append(robot.data.object_pos_w[:, 0, :].reshape(-1, 3).to("cpu"))

            # update buffer
            sim_time = 0.0
            count = 0
            update_buffer_x = np.vstack([touched_buf.clone().cpu().numpy(), untouched_buf.clone().cpu().numpy()])
            update_buffer_y = np.hstack(
                [np.zeros((touched_buf.clone().cpu().numpy().shape[0])),
                 np.ones((untouched_buf.clone().cpu().numpy().shape[0]))])

            # update gpis model and generate estimated_surface
            (uncertainty, xstar,
             estimated_surface, estimated_surface_uncertainty,
             prediction, estimated_surface_prediction,
             miu_gradients, miu_normals, variance_gradients,
             variance_gradients_s, miu_gradients_s, miu_normals_s) = gpis.step(update_buffer_x, update_buffer_y)

            # generation next exploration position

            touched_buf = torch.empty((0, 3), device="cuda")
            untouched_buf = torch.empty((0, 3), device="cuda")

            if len(estimated_surface) > 0:
                Hybrid_control = True

        if not Hybrid_control:
            base_move(robot, controller)

        if Hybrid_control and count % 50 == 0:
            target_pos = move_to_new(robot, controller, target_quat, gpis,
                                     robot.data.object_pos_w[:, 0, :].reshape(-1, 3).to("cpu").numpy(),
                                     entities, entities["sensor"].data.force_matrix_w)
        elif Hybrid_control:
            # move_to_base(robot, controller, entities["sensor"].data.force_matrix_w,
            #              robot.data.object_pos_w[:, 0, :].reshape(-1, 3).to("cpu").numpy(), gpis)
            base_move(robot, controller)

        if count % 5 == 0:
            touched_buf, untouched_buf = store_contact_points_sphere(robot, entities, touched_buf,
                                                                     untouched_buf)

        # write data to sim
        count, sim_time = env_step(ee_marker, goal_marker, robot, robot.data.object_pos_w, robot.data.object_quat_w,
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
