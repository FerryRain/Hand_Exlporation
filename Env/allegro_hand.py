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
from isaaclab_assets import ALLEGRO_HAND_CFG

from ..utils.Controller import Pid_Controller


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


def quat_mul(q1, q2):
    # q1, q2: shape (...,4)
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


def generate_points_with_modified_quaternions(n):
    # 设置参数
    r_total = 5  # 球表面点（半径+0.05）
    center = torch.tensor([0.0, 0.0, -1.0])
    # 为保证全局 z > 0，有：r_total*cos(theta) - 1 > 0  =>  cos(theta) > 1/2.05
    cos_theta_min = 1 / r_total

    # 在 [cos_theta_min, 1] 内均匀采样 cos(theta) 和 phi
    u = torch.empty(n).uniform_(cos_theta_min, 1)  # u = cos(theta)
    theta = torch.acos(u)
    phi = torch.empty(n).uniform_(0, 2 * torch.pi)

    # 球坐标转换（相对于球心）
    sin_theta = torch.sin(theta)
    x = r_total * sin_theta * torch.cos(phi)
    y = r_total * sin_theta * torch.sin(phi)
    z = r_total * u

    # 平移到全局坐标
    points = torch.stack((x, y, z), dim=1) + center

    # 计算每个点的法向量（从球心指向该点，并归一化）
    vec = points - center  # (n,3)
    normals = vec / vec.norm(dim=1, keepdim=True)

    # 参考方向：取 r0 = (0,0,1)，对应 (0,0,2.05) 点的法向量
    r0 = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0).expand(n, 3)

    # 对于每个点，计算从 r0 到 n 的旋转
    # 旋转轴： cross = r0 x normal
    cross = torch.cross(r0, normals, dim=1)
    # 点积（用作计算旋转角）
    dot = (r0 * normals).sum(dim=1).clamp(-1.0, 1.0)  # shape (n,)
    angles = torch.acos(dot)  # 旋转角 θ

    # 计算 sin(θ/2) 和 cos(θ/2)
    sin_half = torch.sin(angles / 2)
    cos_half = torch.cos(angles / 2)

    # 对旋转轴归一化
    axis_norm = cross.norm(dim=1, keepdim=True)
    eps = 1e-6
    # 当 norm 接近 0 时，旋转轴无意义，直接赋值为 0 向量
    axis_normalized = torch.where(axis_norm > eps, cross / axis_norm, torch.zeros_like(cross))

    # 局部旋转四元数 q_local (格式 (x, y, z, w))
    q_local = torch.cat((axis_normalized * sin_half.unsqueeze(1), cos_half.unsqueeze(1)), dim=1)

    # 固定四元数 q_fix，在 (0,0,2.05) 时应为 (0,0,1,0)
    q_fix = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(0).expand(n, 4)

    # 最终的四元数： q = q_fix * q_local
    # 注意乘法顺序： q_fix 在左侧，使得当 q_local 为单位 (0,0,0,1) 时，结果为 q_fix
    quaternions = quat_mul(q_fix, q_local)  # shape (n,4)

    return points, quaternions


def generate_next_exp_points(n):
    # 设置参数
    r_total = 1.2  # 球表面点（半径+0.05）
    center = torch.tensor([0.0, 0.0, -1.0])
    # 为保证全局 z > 0，有：r_total*cos(theta) - 1 > 0  =>  cos(theta) > 1/2.05
    cos_theta_min = 1 / r_total

    # 在 [cos_theta_min, 1] 内均匀采样 cos(theta) 和 phi
    u = torch.empty(n).uniform_(cos_theta_min, 1)  # u = cos(theta)
    theta = torch.acos(u)
    phi = torch.empty(n).uniform_(0, 2 * torch.pi)

    # 球坐标转换（相对于球心）
    sin_theta = torch.sin(theta)
    x = r_total * sin_theta * torch.cos(phi)
    y = r_total * sin_theta * torch.sin(phi)
    z = r_total * u

    # 平移到全局坐标
    points = torch.stack((x, y, z), dim=1) + center

    # 计算每个点的法向量（从球心指向该点，并归一化）
    vec = points - center  # (n,3)
    normals = vec / vec.norm(dim=1, keepdim=True)

    # 参考方向：取 r0 = (0,0,1)，对应 (0,0,2.05) 点的法向量
    r0 = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0).expand(n, 3)

    # 对于每个点，计算从 r0 到 n 的旋转
    # 旋转轴： cross = r0 x normal
    cross = torch.cross(r0, normals, dim=1)
    # 点积（用作计算旋转角）
    dot = (r0 * normals).sum(dim=1).clamp(-1.0, 1.0)  # shape (n,)
    angles = torch.acos(dot)  # 旋转角 θ

    # 计算 sin(θ/2) 和 cos(θ/2)
    sin_half = torch.sin(angles / 2)
    cos_half = torch.cos(angles / 2)

    # 对旋转轴归一化
    axis_norm = cross.norm(dim=1, keepdim=True)
    eps = 1e-6
    # 当 norm 接近 0 时，旋转轴无意义，直接赋值为 0 向量
    axis_normalized = torch.where(axis_norm > eps, cross / axis_norm, torch.zeros_like(cross))

    # 局部旋转四元数 q_local (格式 (x, y, z, w))
    q_local = torch.cat((axis_normalized * sin_half.unsqueeze(1), cos_half.unsqueeze(1)), dim=1)

    # 固定四元数 q_fix，在 (0,0,2.05) 时应为 (0,0,1,0)
    q_fix = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(0).expand(n, 4)

    # 最终的四元数： q = q_fix * q_local
    # 注意乘法顺序： q_fix 在左侧，使得当 q_local 为单位 (0,0,0,1) 时，结果为 q_fix
    quaternions = quat_mul(q_fix, q_local)  # shape (n,4)

    return points, quaternions


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
    allegro_hand = Articulation(ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(0.35, 0.35, 0.035),
            pos=(0., 0., 1.0),
            # rot=(-0.282883,-0.621803,-0.257397,-0.683343),
            joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
        )
    ))

    object_cfg = RigidObject(
        RigidObjectCfg(
            # prim_path="/World/envs/env_.*/object_cfg",
            prim_path="/World/Origin1/object_cfg",
            spawn=sim_utils.SphereCfg(
                # size=(.15, 0.62, 0.3),
                # size=(.2, .62, .2),
                radius=0.1,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True,
                                                             max_linear_velocity=.0, max_angular_velocity=.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=10000),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.0, 1.0, .0)),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
        )
    )

    # return the scene information
    scene_entities = {
        # "allegro": allegro,
        "allegro_hand": allegro_hand,
        # "contact_sensor": con,
        # 'terrain': terrain,
        "object_cfg": object_cfg,
    }
    return scene_entities, origins


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternion to rotation matrix.

    Parameters:
    - quaternions (torch.Tensor): A tensor of shape (N, 4), representing quaternions in the format (w, x, y, z).

    Returns:
    - rotation_matrices (torch.Tensor): A tensor of shape (N, 3, 3), representing the corresponding rotation matrices.
    """
    # Normalize the quaternions
    quaternions = quaternions / torch.linalg.norm(quaternions, dim=1, keepdim=True)

    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Precompute terms for the rotation matrix
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Construct the rotation matrix
    rotation_matrices = torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    ], dim=-2)  # Shape: (N, 3, 3)

    return rotation_matrices


def transform_force_torque_to_local_gpu(force_worlds, torque_worlds, orientation_local_worlds):
    """
    Transform forces and torques from the world frame to the local (body) frame using GPU batch computation.

    Parameters:
    - force_worlds (torch.Tensor): Forces in the world frame, shape (N, 3).
    - torque_worlds (torch.Tensor): Torques in the world frame, shape (N, 3).
    - orientation_local_worlds (torch.Tensor): Quaternions representing the orientation of the local (body) frame
      relative to the world frame, shape (N, 4).

    Returns:
    - force_b (torch.Tensor): Forces in the local (body) frame, shape (N, 3).
    - torque_b (torch.Tensor): Torques in the local (body) frame, shape (N, 3).
    """
    # Ensure input tensors are on the GPU
    force_worlds = force_worlds.to("cuda")
    torque_worlds = torque_worlds.to("cuda")
    orientation_local_worlds = orientation_local_worlds.to("cuda")

    # Convert quaternions to rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(orientation_local_worlds)

    # Compute the transformation matrix from world frame to local frame
    world_to_b_transforms = torch.linalg.inv(rotation_matrices)  # Shape: (N, 3, 3)

    # Transform forces and torques to the local frame
    force_b = torch.einsum("bij,bj->bi", world_to_b_transforms, force_worlds)  # Shape: (N, 3)
    torque_b = torch.einsum("bij,bj->bi", world_to_b_transforms, torque_worlds)  # Shape: (N, 3)

    return force_b, torque_b


def _random_goal(env_ids):
    xy_negative_range = (-0.5, -0.01)
    xy_positive_range = (0.01, 0.5)

    xy_negative_values = torch.rand(len(env_ids), 2) * (xy_negative_range[1] - xy_negative_range[0]) + \
                         xy_negative_range[0]
    xy_positive_values = torch.rand(len(env_ids), 2) * (xy_positive_range[1] - xy_positive_range[0]) + \
                         xy_positive_range[0]
    xy_mask = torch.randint(0, 2, (len(env_ids), 2)).bool()
    xy_values = torch.where(xy_mask, xy_positive_values, xy_negative_values)

    z_negative_range = (-0.5, -0.01)
    z_positive_range = (0.01, 0.5)
    z_negative_values = torch.rand(len(env_ids), 1) * (z_negative_range[1] - z_negative_range[0]) + \
                        z_negative_range[0]
    z_positive_values = torch.rand(len(env_ids), 1) * (z_positive_range[1] - z_positive_range[0]) + \
                        z_positive_range[0]
    z_mask = torch.randint(0, 1, (len(env_ids), 1)).bool()
    z_values = torch.where(z_mask, z_positive_values, z_negative_values)

    random_pos = torch.cat((xy_values, z_values), dim=-1)

    quaternions = torch.randn(len(env_ids), 4)  # 正态分布随机生成
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)  # 单位化

    goals = torch.cat((random_pos, quaternions), dim=1).to("cuda")

    return goals


def move_to(robot, pos, quat):
    print("Exploration new eras")
    robot_state = torch.cat((pos[0, :], quat[0, :]))
    robot_default_state = torch.cat((robot_state, torch.zeros((6))), dim=0).reshape(-1, 13)

    robot.write_root_state_to_sim(robot_default_state)

    goal = True
    if goal:
        print("goal pos: ", pos.to("cpu"))
        print("goal quat: ", quat.to("cpu"))


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str,], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # _terrain = entities['terrain'].class_type(entities['terrain'])
    # Start with hand open
    grasp_mode = 0
    actuated_joint_names = [
        "index_joint_0",
        "middle_joint_0",
        "ring_joint_0",
        "thumb_joint_0",
        "index_joint_1",
        "index_joint_2",
        "index_joint_3",
        "middle_joint_1",
        "middle_joint_2",
        "middle_joint_3",
        "ring_joint_1",
        "ring_joint_2",
        "ring_joint_3",
        "thumb_joint_1",
        "thumb_joint_2",
        "thumb_joint_3",
    ]
    fingertip_body_names = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]

    pid_controller = Pid_Controller(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, device='cuda')

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.1, .1, .1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Simulate physics
    while simulation_app.is_running():
        # reset
        robot = entities['allegro_hand']
        if count % 500 == 0:  # time to next exploration
            # move to next exploration position
            sim_time = 0.0
            count = 0

            pos, quat = generate_next_exp_points(1)
            move_to(robot, pos, quat)
            dof_pos = robot.data.default_joint_pos
            # robot.set_joint_position_target(dof_pos)
            joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_position_to_sim(dof_pos)
            robot.write_joint_state_to_sim(dof_pos, joint_vel)
            i = 1
            # robot_default_state = robot.data.default_root_state.clone()
            # pos, quat = generate_points_with_modified_quaternions(1)
            # # robot_default_state[:, 0:7] = torch.tensor([0., 0., 2.5, 0, 0, 1, 0]).to("cuda")
            # robot_default_state[:, 0:7] = torch.cat((pos[0, :], quat[0, :])).to("cuda").reshape(-1, 7)
            # pos, quat = generate_points_with_modified_quaternions(1)
            # robot.write_root_state_to_sim(robot_default_state)
            # pid_controller.reset(torch.cat((pos[0, :], quat[0, :])).to("cuda").reshape(-1, 7), torch.tensor([0]))
            # print("[INFO]: Resetting robots state...")
        # toggle grasp mode
        # robot_state = robot.data.root_state_w
        # force, torques = pid_controller.step(robot_state)
        # force, torques = transform_force_torque_to_local_gpu(force, torques, robot.data.body_quat_w[:, 0, :])
        # robot.set_external_force_and_torque(forces=force.reshape(-1, 1, 3).to("cuda:0"),
        #                                     torques=torques.reshape(-1, 1, 3).to("cuda:0"), body_ids=[0])
        # if (torch.norm(pid_controller.error_pos, dim=-1) < threshold
        #         and torch.norm(pid_controller.error_rot, dim=-1) < torch.deg2rad(torch.tensor(5))):
        #     print("Goal!")
        else:
            if i == 1:
                grasp_mode = 1

                # generate joint positions
                joint_pos_target = robot.data.soft_joint_pos_limits[..., grasp_mode] - 0.9
                # apply action to the robot
                # robot.set_joint_position_target(joint_pos_target)
                # robot.write_joint_state_to_sim(joint_pos_target, joint_vel)
                i = 0
        # if count % 100 == 0:
        #     grasp_mode = 1 - grasp_mode
        # apply default actions to the hands robots
        # for robot in entities.values():
        #     # generate joint positions
        #     joint_pos_target = robot.data.soft_joint_pos_limits[..., grasp_mode]
        #     # apply action to the robot
        #     robot.set_joint_position_target(joint_pos_target)
        #     # write data to sim
        #     robot.write_data_to_sim()

        # write data to sim
        ee_marker.visualize(robot.data.body_pos_w[0, 0, :].reshape(-1, 3) + origins[0],
                            robot.data.body_quat_w[0, 0, :].reshape(-1, 4))
        # goal_marker.visualize(pid_controller.goal[:, :3] + origins[0],
        #                       pid_controller.goal[:, 3:7])
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
