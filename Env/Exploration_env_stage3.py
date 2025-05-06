"""
@FileName：Exploration_env_stage3.py
@Description：
@Author：Ferry
@Time：2025 4/28/25 2:50 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import argparse

from isaaclab.app import AppLauncher
from utils.HE_GPIS import global_HE_GPIS
from utils.Pid_Controller import Pid_Controller

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


def get_exploration_status(count):
    if count == 250:
        return True
    else:
        return False


def get_grasp_status(count):
    if count == 125:
        return True
    else:
        return False


"""
-------------------------------
# Init origin X and Y positions
-------------------------------
"""


def generate_init_origin_Xy(temp_min, temp_max, grid_count):
    min_data = temp_min - (
            temp_max - temp_min) * 0.2
    max_data = temp_max + (temp_max - temp_min) * 0.2

    x_axis = np.linspace(min_data[0], max_data[0], grid_count)
    y_axis = np.linspace(min_data[1], max_data[1], grid_count)
    z_axis = np.linspace(min_data[2], max_data[2], grid_count)

    origin_X, origin_y = [], []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                origin_X.append([x, y, z])
                origin_y.append(1)

    origin_X = np.array(origin_X)
    origin_y = np.array(origin_y)
    return origin_X, origin_y, min_data, max_data


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    用四元数 q (x, y, z, w) 旋转向量 v (3,)。
    返回旋转后的向量，shape=(3,)
    """
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise ValueError(f"Expect q.shape = (4,), v.shape = (3,), got {q.shape}, {v.shape}")

    qvec = q[:3]  # x, y, z
    q_w = q[3]  # w
    uv = torch.cross(qvec, v)
    uuv = torch.cross(qvec, uv)
    return v + 2.0 * (q_w * uv + uuv)


def quat_from_two_vectors(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """计算将向量 u 旋转到向量 v 的单位四元数"""
    u = u / u.norm()
    v = v / v.norm()
    dot = torch.dot(u, v)

    if torch.isclose(dot, torch.tensor(-1.0, device=u.device)):
        ortho = torch.tensor([1.0, 0.0, 0.0], device=u.device)
        if torch.allclose(u, ortho):
            ortho = torch.tensor([0.0, 1.0, 0.0], device=u.device)
        axis = torch.cross(u, ortho)
        axis = axis / axis.norm()
        return torch.cat([axis * torch.sin(torch.pi / 2), torch.cos(torch.pi / 2).unsqueeze(0)])

    if torch.isclose(dot, torch.tensor(1.0, device=u.device)):
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=u.device)

    axis = torch.cross(u, v)
    axis = axis / axis.norm()
    angle = torch.acos(dot)
    half_angle = angle / 2
    q_xyz = axis * torch.sin(half_angle)
    q_w = torch.cos(half_angle)
    return torch.cat([q_xyz, q_w.unsqueeze(0)])


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数乘法 q_new = q1 * q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w])


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

    goal = True
    if goal:
        # print("goal pos: ", pos.to("cpu"))
        # print("goal quat: ", quat.to("cpu"))
        return True
    else:
        return False


"""
----------------------------
# Store the points to buffer
----------------------------
"""


def add_unique_position_tensor(new_pos, positions, tol=1e-6):
    """
    将 new_pos 添加到 positions (Tensor) 中，若 positions 中已有相近位置则不添加。
    返回更新后的 positions tensor。

    参数:
        new_pos: shape = [3] 的 tensor，需在 GPU 上
        positions: shape = [N, 3] 的 tensor
        tol: 判断重复的容差阈值
    """
    if positions.numel() == 0:
        return new_pos.unsqueeze(0)  # 第一个点，初始化

    # 计算所有现有点和 new_pos 的 L2 距离
    dists = torch.norm(positions - new_pos.unsqueeze(0), dim=1)

    if torch.any(dists < tol):
        return positions  # 已存在类似的点，跳过添加

    # 添加新点
    return torch.cat([positions, new_pos.unsqueeze(0)], dim=0)


def store_contact_points(robot, entities, touched_buffer, untouched_buffer):
    if torch.norm(entities["index_sensor"].data.force_matrix_w, dim=-1) > 0:
        touched_buffer = add_unique_position_tensor(entities["index_sensor"].data.pos_w.reshape(-1), touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["index_sensor"].data.pos_w.reshape(-1), touched_buffer)

    if torch.norm(entities["middle_sensor"].data.force_matrix_w, dim=-1) > 0:
        touched_buffer = add_unique_position_tensor(entities["middle_sensor"].data.pos_w.reshape(-1), touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["middle_sensor"].data.pos_w.reshape(-1),
                                                      touched_buffer)

    if torch.norm(entities["ring_sensor"].data.force_matrix_w, dim=-1) > 0:
        touched_buffer = add_unique_position_tensor(entities["ring_sensor"].data.pos_w.reshape(-1),
                                                    touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["ring_sensor"].data.pos_w.reshape(-1),
                                                      touched_buffer)
    if torch.norm(entities["thumb_sensor"].data.force_matrix_w, dim=-1) > 0:
        touched_buffer = add_unique_position_tensor(entities["thumb_sensor"].data.pos_w.reshape(-1),
                                                    touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["thumb_sensor"].data.pos_w.reshape(-1),
                                                      touched_buffer)

    return touched_buffer, untouched_buffer


"""
-------------------------------
# Generate next exploration point
-------------------------------
"""
from scipy.spatial.transform import Rotation as R

def is_normalized_quat(q, tol=1e-3):
    norm = torch.linalg.norm(q)
    return torch.abs(norm - 1.0) < tol


def is_valid_quat(q):
    return not torch.isnan(q).any() and not torch.isinf(q).any()


def check_quat_validity(q, tol=1e-8):
    if q.shape[-1] != 4:
        return None
    if not is_valid_quat(q):
        return None
    if not is_normalized_quat(q, tol):
        # print("Warning: Quaternion not normalized. Normalizing automatically.")
        q = q / torch.linalg.norm(q)
    return q.reshape(-1, 4)


def kd_pruning(surface_points, uncertainty, prediction, radius=0.05):
    tree = KDTree(surface_points)

    selected_indices = []
    visited = np.zeros(len(surface_points), dtype=bool)

    for i in range(len(surface_points)):
        if visited[i]:
            continue
        idx = tree.query_radius([surface_points[i]], r=radius)[0]
        visited[idx] = True
        selected_indices.append(i)

    # Keep only diverse points
    surface_points_pruned = surface_points[selected_indices]
    uncertainty_pruned = uncertainty[selected_indices]
    prediction_pruned = prediction[selected_indices]

    return surface_points_pruned, uncertainty_pruned, prediction_pruned


def kd_cut_down(surface_points, explored_queue, uncertainty, prediction, k=10):
    tree = KDTree(surface_points)
    index = []

    for p in explored_queue:
        dist, idx = tree.query([p], k=k)
        index.extend(idx[0])  # idx is a 2D array

    mask = np.ones(len(surface_points), dtype=bool)
    mask[index] = False

    surface_points_cutdown = surface_points[mask]
    uncertainty_cutdown = uncertainty[mask]
    prediction_cutdown = prediction[mask]
    return surface_points_cutdown, uncertainty_cutdown, prediction_cutdown


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def calculate_points_score(
            origin_points, surface_points, uncertainty, prediction, hand_pos_now, explored_queue,
            alpha=.5, beta=1.0, gamma=.5, delta=2.0, epsilon=0.05,
            proximity_radius=0.05
        ):
    # Uncertainty score U(x)
    U = uncertainty
    # Distance penalty D(x)
    D = -np.linalg.norm(origin_points - hand_pos_now.squeeze(), axis=1)  # (1401,)
    # Information score I(x)
    I = np.exp(-np.abs(prediction) / epsilon) * uncertainty * 1e10 # (1401,)

    U_norm = normalize(U)
    D_norm = normalize(D)
    I_norm = normalize(I)

    S = alpha * U_norm + beta * D_norm + gamma * I_norm

    tree_explored = KDTree(explored_queue)
    distances, _ = tree_explored.query(origin_points, k=1)  # closest explored point per surface_point
    P = np.where(distances.squeeze() < proximity_radius, -1.0, 0.0)  # (1401,)
    S = S + delta * P

    return S


def align_hand_palm(quat_current, hand_normal):
    """
    quat_current: np.array (4,), (x, y, z, w) 当前手的四元数
    hand_normal: np.array (3,) 单位向量，目标掌心朝向
    return: quat_target (4,), (x, y, z, w)
    """
    # Step 1: 当前掌心朝向
    r_current = R.from_quat(quat_current)
    rotation_matrix = r_current.as_matrix()  # (3, 3)

    # 手掌局部 z+ 方向在世界系下的朝向
    current_palm_dir = rotation_matrix[:, 2]  # 第三列，局部 z+

    # Step 2: 归一化
    current_palm_dir /= (np.linalg.norm(current_palm_dir) + 1e-8)
    target_dir = hand_normal / (np.linalg.norm(hand_normal) + 1e-8)

    # Step 3: 计算旋转轴 和 旋转角
    axis = np.cross(current_palm_dir, target_dir)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        if np.dot(current_palm_dir, target_dir) > 0:
            rotation_quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
        else:
            # 180度旋转，找一个垂直方向 (比如 x 轴)
            rotation_quat = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_quat()
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(current_palm_dir, target_dir), -1.0, 1.0))
        rotation_quat = R.from_rotvec(axis * angle).as_quat()  # (x,y,z,w)

    # Step 4: 补偿旋转 * 当前手的旋转
    r_rotation = R.from_quat(rotation_quat)
    r_target = r_rotation * r_current  # 注意顺序：补偿旋转 × 当前旋转
    quat_target = r_target.as_quat()  # (x, y, z, w)

    return quat_target



def calculate_rot(target_pos, surface_points, hand_quat):
    tree_surface = KDTree(surface_points)
    distance, index = tree_surface.query([target_pos], k=1)
    nearest_surface_point = surface_points[index[0][0]]  # (3,)

    v = nearest_surface_point - target_pos  # (3,)
    hand_normal = v / (np.linalg.norm(v) + 1e-8)

    quat_target = align_hand_palm(hand_quat, hand_normal)
    return quat_target

def sample_next_exploration_point_surface(surface_points, uncertainty, prediction, hand_pos_now, explored_queue, hand_quat):
    explored_queue_np = np.array(explored_queue).reshape(-1, 3)
    surface_points_pruned, uncertainty_pruned, prediction_pruned = kd_pruning(surface_points, uncertainty, prediction)
    surface_points_cutdown, uncertainty_cutdown, prediction_cutdown = kd_cut_down(
        surface_points_pruned,
        explored_queue_np,
        uncertainty_pruned,
        prediction_pruned
    )

    S = calculate_points_score(surface_points_cutdown, surface_points_cutdown, uncertainty_cutdown, prediction_cutdown, hand_pos_now,
                               explored_queue_np)

    best_idx = np.argmax(S)
    pos_target = surface_points[best_idx]

    quat_target =calculate_rot(pos_target, surface_points_pruned, hand_quat)

    return (torch.tensor(pos_target, dtype=torch.float32, device='cuda').reshape(-1, 3),
            torch.tensor(quat_target, dtype=torch.float32, device='cuda').reshape(-1, 4))

def sample_next_exploration_point(origin, surface_points, uncertainty, prediction, estimated_surface_uncertainty, etimated_surface_prediction, hand_pos_now, explored_queue, hand_quat):
    explored_queue_np = np.array(explored_queue).reshape(-1, 3)
    surface_points_pruned, uncertainty_pruned, prediction_pruned = kd_pruning(surface_points, estimated_surface_uncertainty, etimated_surface_prediction)
    origin_cutdown, uncertainty_cutdown, prediction_cutdown = kd_cut_down(
        origin,
        explored_queue_np,
        uncertainty,
        prediction,
        k=100
    )

    S = calculate_points_score(origin_cutdown, surface_points_pruned, uncertainty_cutdown, prediction_cutdown, hand_pos_now,
                               explored_queue_np)

    best_idx = np.argmax(S)
    pos_target = origin[best_idx]

    quat_target =calculate_rot(pos_target, surface_points_pruned, hand_quat)

    return (torch.tensor(pos_target, dtype=torch.float32, device='cuda').reshape(-1, 3),
            torch.tensor(quat_target, dtype=torch.float32, device='cuda').reshape(-1, 4))




def run_simulator(sim: sim_utils.SimulationContext, entities):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Start with hand open
    grasp = True
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

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (.1, .1, .1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot = entities['robot']
    alpha = 0.05
    pos = torch.tensor([0, .2, -0.48], device="cuda")
    quat = torch.tensor((0.257551, 0.283045, 0.683330, -0.621782), device="cuda")
    grasp_joint_pos = robot.data.default_joint_pos.clone()

    target_pos, target_quat = pos.reshape(-1, 3) + entities.env_origins[0], quat.reshape(-1, 4)

    touched_buffer = torch.empty((0, 3), device="cuda")
    untouched_buffer = torch.empty((0, 3), device="cuda")

    res = 100  # 150 # grid resolution # 50

    display_percentile_low = 10
    display_percentile_high = 90
    training_iter = 200
    grid_count = 10

    temp_min = np.array([-0.42, -0.42, -0.42])
    temp_max = np.array([0.42, 0.42, 0.42])
    origin_X, origin_y, min_data, max_data = generate_init_origin_Xy(temp_min, temp_max, grid_count)

    # init gpis model
    gpis = global_HE_GPIS(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, show_points=True,
        store=True, store_path="../Data/Exploration_env_stage3_"
    )

    K_p_pos, K_i_pos, K_d_pos = 10.0, 0.0001, 7.0  # pos p i d of  PID
    K_p_rot, K_i_rot, K_d_rot = 0.5, 0.025, 0.65  # rot p i d of  PID
    dt = 0.01
    threshold = 0.01
    threshold_ang = 5

    # init Hand position Controller (local control in isaaclab)
    controller = Pid_Controller(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, local_control=True)

    # Initialize a queue store the
    explored_queue = deque(maxlen=10)

    # Simulate physics
    while simulation_app.is_running():

        if get_exploration_status(count):  # time to next exploration
            explored_queue.append(robot.data.body_pos_w[0, 0, :].reshape(-1, 3).to("cpu"))
            # move to next exploration position
            sim_time = 0.0
            count = 0
            update_buffer_x = np.vstack([touched_buffer.cpu().numpy(), untouched_buffer.cpu().numpy()])
            update_buffer_y = np.hstack(
                [np.zeros((touched_buffer.cpu().numpy().shape[0])), np.ones((untouched_buffer.cpu().numpy().shape[0]))])

            # update gpis model and generate estimated_surface
            uncertainty, xstar, estimated_surface, estimated_surface_uncertainty, prediction, estimated_surface_prediction = gpis.step(
                update_buffer_x, update_buffer_y)

            # target_pos, target_quat = pos.reshape(-1,3) + entities.env_origins[0], quat.reshape(-1,4)
            if len(estimated_surface) > 0:
                target_pos, target_quat = sample_next_exploration_point(
                    xstar,
                    estimated_surface,
                    uncertainty,
                    prediction,
                    estimated_surface_uncertainty,
                    estimated_surface_prediction,
                    robot.data.body_pos_w[0, 0,:].reshape(-1, 3).to("cpu").numpy(),
                    explored_queue,
                    robot.data.body_quat_w[0, 0,:].reshape(-1, 4).to("cpu").numpy()
                )
                target_pos = target_pos + entities.env_origins[0]
                target_quat_check = check_quat_validity(target_quat)
                if target_quat_check is not None:
                    target_quat = target_quat_check

            grasp_joint_pos = robot.data.default_joint_pos.clone()

            touched_buffer = torch.empty((0, 3), device="cuda")
            untouched_buffer = torch.empty((0, 3), device="cuda")


        elif get_grasp_status(count):
            # grasp
            if grasp == True:
                # generate joint positions
                grasp_joint_pos = robot.data.soft_joint_pos_limits[..., 1] - 0.9

        pre_joint_pos_target = (1 - alpha) * robot.data.joint_pos + alpha * grasp_joint_pos
        robot.set_joint_position_target(pre_joint_pos_target)

        grasp = move_to(robot, controller, target_pos, target_quat)

        if count % 50 == 0:
            touched_buffer, untouched_buffer = store_contact_points(robot, entities, touched_buffer, untouched_buffer)

        # write data to sim
        ee_marker.visualize(robot.data.body_pos_w[0, 0, :].reshape(-1, 3) + entities.env_origins[0],
                            robot.data.body_quat_w[0, 0, :].reshape(-1, 4))
        goal_marker.visualize(target_pos + entities.env_origins[0],
                              target_quat)

        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


if __name__ == '__main__':
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.2, 0.5])
    # design scene
    scene_cfg = ExpllorationEnvCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    scene = InteractiveScene(scene_cfg)
    # scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)
    simulation_app.close()
