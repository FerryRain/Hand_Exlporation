"""
@FileName：Exploration_env_stage2.py
@Description：
@Author：Ferry
@Time：2025 4/27/25 1:50 PM
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
from isaaclab_assets import ALLEGRO_HAND_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


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
    if count == 500:
        return True
    else:
        return False


def get_grasp_status(count):
    if count == 250:
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
from sklearn.neighbors import NearestNeighbors

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
    return q.reshape(-1,4)

def estimate_normal(surface: np.ndarray, index: int, k=20):
    nbrs = NearestNeighbors(n_neighbors=k).fit(surface)
    _, indices = nbrs.kneighbors([surface[index]])

    neighbors = surface[indices[0]]
    pca = np.cov(neighbors.T)
    _, _, vh = np.linalg.svd(pca)
    normal = vh[-1]  # 法向量是最小特征值对应的方向
    return torch.tensor(normal, dtype=torch.float32, device='cuda')

def generate_hand_pose_outside_surface(
        surface: np.ndarray,
        q_default: torch.Tensor,
        hand_palm_dir_local: torch.Tensor,
        offset: float = 0.05,
        device: str = 'cuda'
):
    # 1. 采样表面点
    idx = np.random.randint(len(surface))
    p = surface[idx]
    p_torch = torch.tensor(p, dtype=torch.float32, device=device)

    # 2. 估算法向量（局部邻域 PCA）
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=20).fit(surface)
    _, indices = nbrs.kneighbors([p])
    neighbors = surface[indices[0]]
    cov = np.cov(neighbors.T)
    _, _, vh = np.linalg.svd(cov)
    n = vh[-1]  # 法向量（最小主成分方向）
    n = torch.tensor(n, dtype=torch.float32, device=device)

    # 3. 生成手的位置（往外移动）
    position = p_torch - offset * n  # 法向量反方向

    # 4. 当前手心朝向（世界坐标下）
    palm_dir_world = quat_rotate(q_default, hand_palm_dir_local)

    # 5. 构造旋转对齐
    q_align = quat_from_two_vectors(palm_dir_world, n)

    # 6. 新四元数
    q_new = quat_mul(q_align, q_default)

    return position, q_new

def sample_next_exploration_point(estimated_surface, surface_uncertainty, strategy="max", temperature=0.1):
    """
    支持两种策略:
    - "max": 选择不确定性最大的点
    - "softmax": 以不确定性为权重做softmax采样（加点随机性，避免陷入局部）
    """

    if strategy == "max":
        max_idx = np.argmax(surface_uncertainty)
        next_point = estimated_surface[max_idx]
    elif strategy == "softmax":
        probs = np.exp(surface_uncertainty / temperature)
        probs /= probs.sum()
        idx = np.random.choice(len(estimated_surface), p=probs)
        next_point = estimated_surface[idx]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return torch.from_numpy(next_point).float().to('cuda').reshape(-1, 3)



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

    temp_min = np.array([-0.5, -0.5, -0.5])
    temp_max = np.array([0.5, 0.5, 0.5])
    origin_X, origin_y, min_data, max_data = generate_init_origin_Xy(temp_min, temp_max, grid_count)

    # init gpis model
    gpis = global_HE_GPIS(res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
                          origin_y,
                          min_data, max_data, show_points=True)

    K_p_pos, K_i_pos, K_d_pos = 10.0, 0.0001, 7.0  # pos p i d of  PID
    K_p_rot, K_i_rot, K_d_rot = 0.5, 0.025, 0.65  # rot p i d of  PID
    dt = 0.01
    threshold = 0.01
    threshold_ang = 5

    # init Hand position Controller (local control in isaaclab)
    controller = Pid_Controller(K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, 1, dt, local_control=True)

    # Simulate physics
    while simulation_app.is_running():

        if get_exploration_status(count):  # time to next exploration
            # move to next exploration position
            sim_time = 0.0
            count = 0
            update_buffer_x = np.vstack([touched_buffer.cpu().numpy(), untouched_buffer.cpu().numpy()])
            update_buffer_y = np.hstack(
                [np.zeros((touched_buffer.cpu().numpy().shape[0])), np.ones((untouched_buffer.cpu().numpy().shape[0]))])

            # update gpis model and generate estimated_surface
            uncertainty, xstar, estimated_surface, estimated_surface_uncertainty = gpis.step(update_buffer_x, update_buffer_y)

            # target_pos, target_quat = pos.reshape(-1,3) + entities.env_origins[0], quat.reshape(-1,4)
            if len(estimated_surface) > 0:
                # target_pos = sample_next_exploration_point(estimated_surface, estimated_surface_uncertainty,
                #                                            strategy="softmax")  + entities.env_origins[0]
                target_pos = sample_next_exploration_point(xstar, uncertainty,
                                                           strategy="softmax") + entities.env_origins[0]
                target_quat = torch.tensor((-0.0180, 0.3823, 0.9229, 0.0435), device="cuda").reshape(-1,4)
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

        if count % 50 ==0:
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
