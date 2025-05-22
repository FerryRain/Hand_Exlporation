"""
@FileName：utils.py
@Description：
@Author：Ferry
@Time：2025 5/6/25 2:42 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from utils.HE_GPIS import global_HE_GPIS, normal_HE_GPIS, normal_HE_GPIS_2, normal_HE_GPIS_3, normal_HE_GPIS_4

"""
-------------------------------
# Exploration policy move state
-------------------------------
"""


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
    # min_data = temp_min - (
    #         temp_max - temp_min) * 0.2
    # max_data = temp_max + (temp_max - temp_min) * 0.2

    x_axis = np.linspace(temp_min[0], temp_max[0], grid_count)
    y_axis = np.linspace(temp_min[1], temp_max[1], grid_count)
    z_axis = np.linspace(temp_min[2], temp_max[2], grid_count)

    origin_X, origin_y = [], []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                origin_X.append([x, y, z])
                origin_y.append(1)

    origin_X = np.array(origin_X)
    origin_y = np.array(origin_y)
    return origin_X, origin_y, temp_min, temp_max


def generate_init_origin_Xy_layered(outer_min, outer_max, inner_min, inner_max, grid_count):
    x_axis = np.linspace(outer_min[0], outer_max[0], grid_count)
    y_axis = np.linspace(outer_min[1], outer_max[1], grid_count)
    z_axis = np.linspace(outer_min[2], outer_max[2], grid_count)

    origin_X = []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                if (inner_min[0] <= x <= inner_max[0] and
                        inner_min[1] <= y <= inner_max[1] and
                        inner_min[2] <= z <= inner_max[2]):
                    continue
                origin_X.append([x, y, z])

    origin_X = np.array(origin_X)
    origin_y = np.ones(len(origin_X))
    return origin_X, origin_y, outer_min, outer_max, inner_min, inner_max


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
----------------------------
# Contact sensors
----------------------------
"""


def contact_detect(froce):
    if torch.norm(froce, dim=-1) > 0:
        return True
    else:
        return False


def add_unique_position_tensor(new_pos, positions, tol=1e-2):
    if positions.numel() == 0:
        return new_pos.unsqueeze(0)

    dists = torch.norm(positions - new_pos.unsqueeze(0), dim=1)

    if torch.any(dists < tol):
        return positions

    # 添加新点
    return torch.cat([positions, new_pos.unsqueeze(0)], dim=0)


def store_contact_points(robot, entities, touched_buffer, untouched_buffer):
    if contact_detect(entities["index_sensor"].data.force_matrix_w):
        touched_buffer = add_unique_position_tensor(entities["index_sensor"].data.pos_w.reshape(-1), touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["index_sensor"].data.pos_w.reshape(-1), untouched_buffer)

    if contact_detect(entities["middle_sensor"].data.force_matrix_w):
        touched_buffer = add_unique_position_tensor(entities["middle_sensor"].data.pos_w.reshape(-1), touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["middle_sensor"].data.pos_w.reshape(-1),
                                                      untouched_buffer)

    if contact_detect(entities["ring_sensor"].data.force_matrix_w):
        touched_buffer = add_unique_position_tensor(entities["ring_sensor"].data.pos_w.reshape(-1),
                                                    touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["ring_sensor"].data.pos_w.reshape(-1),
                                                      untouched_buffer)
    if contact_detect(entities["thumb_sensor"].data.force_matrix_w):
        touched_buffer = add_unique_position_tensor(entities["thumb_sensor"].data.pos_w.reshape(-1),
                                                    touched_buffer)
    else:
        untouched_buffer = add_unique_position_tensor(entities["thumb_sensor"].data.pos_w.reshape(-1),
                                                      untouched_buffer)

    return touched_buffer, untouched_buffer


def store_contact_points_sphere(robot, entities, touched_buffer, untouched_buffer):
    if contact_detect(entities["sensor"].data.force_matrix_w):
        touched_buffer = add_unique_position_tensor(entities["sensor"].data.pos_w.reshape(-1), touched_buffer.clone())
        # touched_buffer = torch.cat([touched_buffer, entities["sensor"].data.pos_w.reshape(-1).unsqueeze(0)], dim=0)
    else:
        untouched_buffer = add_unique_position_tensor(entities["sensor"].data.pos_w.reshape(-1),
                                                      untouched_buffer.clone())

    return touched_buffer, untouched_buffer


"""
-------------------------------
# Generate next exploration point
-------------------------------
"""


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-14)


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


def kd_nearest(points, pos):
    tree = KDTree(points)

    dist, idx = tree.query([pos[0]], k=1)

    return idx[0]


def kd_pruning(points, radius=0.05):
    tree = KDTree(points)

    selected_indices = []
    visited = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if visited[i]:
            continue
        idx = tree.query_radius([points[i]], r=radius)[0]
        visited[idx] = True
        selected_indices.append(i)

    return selected_indices


def kd_cut_down(points, cut_down_queue, uncertainty, prediction, k=10):
    tree = KDTree(points)
    index = []
    k = min(k, len(points))
    for p in cut_down_queue:
        dist, idx = tree.query([p], k=k)
        index.extend(idx[0])  # idx is a 2D array

    mask = np.ones(len(points), dtype=bool)
    mask[index] = False

    points_cutdown = points[mask]
    uncertainty_cutdown = uncertainty[mask]
    prediction_cutdown = prediction[mask]
    return points_cutdown, uncertainty_cutdown, prediction_cutdown


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

    quat_target = align_hand_palm(hand_quat, -hand_normal)
    return quat_target


def compute_projected_unit_direction_np(grad, normal):
    """
    d_i = (P_n(x) * grad) / ||P_n(x) * grad||_2
    """
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    P_n = np.eye(3) - np.outer(normal, normal)
    projected = P_n @ grad.reshape(3, )

    norm = np.linalg.norm(projected) + 1e-10
    d_i = projected / norm

    return d_i.reshape(-1, 3)


"""
-------------------------------
# Setup Env state
-------------------------------
"""


def env_step(ee_marker, goal_marker, robot, body_pos_w, body_quat_w, entities, target_pos, target_quat, sim, sim_time,
             sim_dt, count, scene):
    # write data to sim
    ee_marker.visualize(body_pos_w[:, 0, :].reshape(-1, 3) + entities.env_origins[0],
                        body_quat_w[:, 0, :].reshape(-1, 4))
    goal_marker.visualize(target_pos + entities.env_origins[0],
                          target_quat)
    robot.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    count += 1
    scene.update(sim_dt)
    return count, sim_time


"""
-------------------------------
# init GPIS Model
-------------------------------
"""


def init_global_HE_GPIS_model(temp_min, temp_max, res=100, display_percentile_low=10, display_percentile_high=90,
                              training_iter=200, grid_count=10, store_path="../Results/Exploration_env_stage3_"):
    origin_X, origin_y, min_data, max_data = generate_init_origin_Xy(temp_min, temp_max, grid_count)

    # init gpis model
    gpis = global_HE_GPIS(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, show_points=True,
        store=True, store_path=store_path
    )

    return gpis


def init_normal_HE_GPIS_model(temp_min, temp_max, res=100, display_percentile_low=10, display_percentile_high=90,
                              training_iter=200, grid_count=10, store_path="../Results/Exploration_env_stage5_"):
    origin_X, origin_y, min_data, max_data = generate_init_origin_Xy(temp_min, temp_max, grid_count)

    # init gpis model
    gpis = normal_HE_GPIS(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, show_points=True,
        store=True, store_path=store_path
    )

    return gpis


def init_normal_HE_GPIS_model_2(temp_min, temp_max, bb_min, bb_max, res=100, display_percentile_low=10,
                                display_percentile_high=90,
                                training_iter=200, grid_count=10, store_path="../Results/Exploration_env_stage5_"):
    origin_X, origin_y, min_data, max_data, bbmin, bbmax = generate_init_origin_Xy_layered(temp_min, temp_max, bb_min,
                                                                                           bb_max, grid_count)

    # init gpis model
    gpis = normal_HE_GPIS_2(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, bbmin, bbmax, show_points=True,
        store=True, store_path=store_path
    )

    return gpis

def init_normal_HE_GPIS_model_3(temp_min, temp_max, bb_min, bb_max, res=100, display_percentile_low=10,
                                display_percentile_high=90,
                                training_iter=200, grid_count=10, store_path="../Results/Exploration_env_stage5_"):
    origin_X, origin_y, min_data, max_data, bbmin, bbmax = generate_init_origin_Xy_layered(temp_min, temp_max, bb_min,
                                                                                           bb_max, grid_count)

    # init gpis model
    gpis = normal_HE_GPIS_3(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, bbmin, bbmax, show_points=True,
        store=True, store_path=store_path
    )

    return gpis

def init_normal_HE_GPIS_model_4(temp_min, temp_max, bb_min, bb_max, res=100, display_percentile_low=10,
                                display_percentile_high=90,
                                training_iter=200, grid_count=10, store_path="../Results/Exploration_env_stage5_"):
    origin_X, origin_y, min_data, max_data, bbmin, bbmax = generate_init_origin_Xy_layered(temp_min, temp_max, bb_min,
                                                                                           bb_max, grid_count)

    # init gpis model
    gpis = normal_HE_GPIS_4(
        res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
        origin_y,
        min_data, max_data, bbmin, bbmax, show_points=True,
        store=True, store_path=store_path
    )

    return gpis


"""
-------------------------------
# Compute Uncertainty Measure
-------------------------------
"""


def compute_mean_surface_uncertainty(variance, area_weights):
    """
    U(D) = sum V(x_i) * A_i / sum A_i
    """
    numerator = np.sum(variance * area_weights)
    denominator = np.sum(area_weights)
    return numerator / (denominator + 1e-8)
