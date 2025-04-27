"""
@FileName：Pid_Controller.py
@Description：
@Author：Ferry
@Time：2024/12/17 下午5:11
@Copyright：©2024-2024 ShanghaiTech University-RIMLAB
"""
import torch

"""
self.error_integral_pos = torch.zeros((num_envs, 3), device=device)
        self.error_integral_rot = torch.zeros((num_envs, 3), device=device)
        self.goal = torch.tensor([[0.5, 0.5, 0.5, 1, 0, 0, 0]] * num_envs, device=device)
"""

class Pid_Controller:
    def __init__(self, K_p_pos, K_i_pos, K_d_pos, K_p_rot, K_i_rot, K_d_rot, num_envs, dt=0.01, device='cuda', local_control=True):
        self.K_p_pos = K_p_pos
        self.K_i_pos = K_i_pos
        self.K_d_pos = K_d_pos
        self.K_p_rot = K_p_rot
        self.K_i_rot = K_i_rot
        self.K_d_rot = K_d_rot
        self.dt = dt
        self.device = device
        self.error_integral_pos = torch.zeros((num_envs, 3), device=device)
        self.error_integral_rot = torch.zeros((num_envs, 3), device=device)
        self.goal = torch.tensor([[0.5, 0.5, 0.5, 1, 0, 0, 0]] * num_envs, device=device)

        self.local_control = local_control

    def reset(self, goal, env_ids):
        """
        Reset the PID controller's internal state and set a new target.
        Input:
            - goal: The new target point (batch_size, 7), where the first 3 values are the position,
                    and the last 4 values represent the target quaternion.
        """
        self.error_integral_pos[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.error_integral_rot[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self.goal[env_ids] = goal[env_ids]

    def _quaternion_conjugate_t(self, q):
        """
        Compute the conjugate of a batch of quaternions.
        Input: q (batch_size, 4) -> [w, x, y, z]
        Output: Conjugate quaternions (batch_size, 4)
        """
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def _quaternion_multiply_t(self, q1, q2):
        """
        Compute the product of two quaternions in a batch.
        Input: q1, q2 (batch_size, 4) -> [w, x, y, z]
        Output: q1 ⊗ q2 (batch_size, 4)
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dim=-1)

    def _compute_error_quaternion(self, target_q, current_q):
        """
        Compute the error quaternion between the target and current quaternions.
        Input:
            - target_q: Target quaternion (batch_size, 4)
            - current_q: Current quaternion (batch_size, 4)
        Output:
            - error_q: Error quaternion (batch_size, 4)
        """
        # Conjugate of the current quaternion
        current_q_conj = self._quaternion_conjugate_t(current_q)
        # Compute the error quaternion
        error_q = self._quaternion_multiply_t(target_q, current_q_conj)
        return error_q

    def _quaternion_to_axis_angle_t(self, q):
        """
        Convert a batch of quaternions to rotation axes and angles.
        Input: q (batch_size, 4) -> [w, x, y, z]
        Output:
            - axis (batch_size, 3): Rotation axis
            - angle (batch_size,): Rotation angle
        """
        w, x, y, z = q.unbind(-1)
        angle = 2.0 * torch.arccos(w.clamp(-1.0, 1.0))  # Compute rotation angle
        sin_half_angle = torch.sqrt((1.0 - w ** 2) + 1e-6).clamp(min=1e-8)  # Prevent division by zero
        axis = torch.stack([x, y, z], dim=-1) / sin_half_angle.unsqueeze(-1)
        return axis, angle.unsqueeze(-1)

    def quaternion_to_rotation_matrix(self, quaternions):
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

    def transform_force_torque_to_local_gpu(self, force_worlds, torque_worlds, orientation_local_worlds):
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
        rotation_matrices = self.quaternion_to_rotation_matrix(orientation_local_worlds)

        # Compute the transformation matrix from world frame to local frame
        world_to_b_transforms = torch.linalg.inv(rotation_matrices)  # Shape: (N, 3, 3)

        # Transform forces and torques to the local frame
        force_b = torch.einsum("bij,bj->bi", world_to_b_transforms, force_worlds)  # Shape: (N, 3)
        torque_b = torch.einsum("bij,bj->bi", world_to_b_transforms, torque_worlds)  # Shape: (N, 3)

        return force_b, torque_b

    def step(self, robot_state):
        """
        Compute the PID forces and torques for position and orientation control.
        Input:
            - robot_state: Current robot state (batch_size, 13), where:
                - First 3 values are the current position.
                - Next 4 values are the current quaternion.
                - Next 3 values are the linear velocity.
                - Last 3 values are the angular velocity.
        Output:
            - force: Control forces for position (batch_size, 3).
            - torques: Control torques for orientation (batch_size, 3).
        """
        # Extract target and current states
        target_position = self.goal[:, :3]
        target_quaternion = self.goal[:, 3:7]

        current_position = robot_state[:, 0:3]
        current_quaternion = robot_state[:, 3:7]
        velocity = robot_state[:, 7:10]
        angular_velocity = robot_state[:, 10:13]

        # Compute position control
        self.error_pos = target_position - current_position
        self.error_integral_pos[:] += self.error_pos * self.dt
        force = (
                self.K_p_pos * self.error_pos +
                self.K_i_pos * self.error_integral_pos -
                self.K_d_pos * velocity
        )

        # Compute orientation control
        axis, angle = self._quaternion_to_axis_angle_t(
            self._compute_error_quaternion(target_quaternion, current_quaternion)
        )
        self.error_rot = angle * axis
        self.error_integral_rot[:] += self.error_rot * self.dt

        # PID control formula (rotation)
        torques = (
                self.K_p_rot * self.error_rot +
                self.K_i_rot * self.error_integral_rot -
                self.K_d_rot * angular_velocity
        )
        if self.local_control:
            force, torques = self.transform_force_torque_to_local_gpu(force, torques, current_quaternion)
            return force, torques
        else:
            return force, torques
