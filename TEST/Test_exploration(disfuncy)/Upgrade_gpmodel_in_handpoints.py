"""
@FileName：Upgrade_gpmodel_in_handpoints.py
@Description：
@Author：Ferry
@Time：3/26/25 3:35 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import time

import torch
import gpytorch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from Hand_vertices import Hand_fingers_vertices



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ObjectShapeGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ObjectShapeGP, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def generate_initial_cube():
    x = np.linspace(-1e-2, 1e-2, 10)
    y = np.linspace(-1e-2, 1e-2, 10)
    z = np.linspace(-1e-2, 1e-2, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    return np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T


def explore_and_update_gp(model, likelihood, train_x, train_y):
    sampled_points = hand.sampled_tensor.to(device)
    observed_values = torch.ones(hand.sampled_tensor.shape[0]).to(device)
    train_x = torch.cat([train_x, sampled_points], dim=0)
    train_y = torch.cat([train_y, observed_values], dim=0)
    train_x_pre = train_x.clone()
    train_y_pre = smooth(train_x, train_y)
    # 重新训练 GP
    model.set_train_data(train_x_pre, train_y_pre, strict=False)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(500):
        optimizer.zero_grad()
        output = model(train_x_pre)
        loss = -mll(output, train_y_pre.nan_to_num(0.0))
        loss.backward()
        optimizer.step()

    return model, train_x, train_y_pre

def smooth(points, value, sigma=0.005, thresh = 0.5):
    mask_one = value == 1
    one_points = points[mask_one]

    distances = torch.cdist(points, one_points)
    min_dist, _ = distances.min(dim=1)
    # smooth_values = (0.9 * value + 0.1 * torch.exp(-min_dist ** 2 / (2 * sigma ** 2))).clamp(max=1)
    smooth_values = torch.exp(-min_dist ** 2 / (2 * sigma ** 2))

    # 如果你希望原本为 0 的点也平滑，不妨直接使用 smooth_values
    # 而如果想保留原始标签为1的点固定为1，可以做如下处理：
    smooth_values_final = smooth_values.clone()
    smooth_values_final[mask_one] = 1.0
    return smooth_values_final

if __name__ == '__main__':
    train_points = torch.tensor(generate_initial_cube(), dtype=torch.float32).to(device)
    train_values = torch.zeros((train_points.shape[0],), dtype=torch.float32).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ObjectShapeGP(train_points, train_values, likelihood).to(device)

    hand = Hand_fingers_vertices(grasp_code="core-bottle-cb3ff04a607ea9651b22d29e47ec3f2.npy")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    hand.update(5)
    for i in range(2000):
        model, train_points, train_values = explore_and_update_gp(model, likelihood, train_points, train_values)

        hand.update(5)
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            pred_dist = model(train_points)
            mean = pred_dist.mean.cpu().numpy()
        train_values = pred_dist.mean
        ax.clear()
        mask = mean > 0.5
        train_points_cpu = train_points.cpu().numpy()
        ax.scatter(train_points_cpu[:, 0][mask],
                   train_points_cpu[:, 1][mask],
                   train_points_cpu[:, 2][mask],
                   c=mean[mask], marker='o')
        ax.set_xlim([-5e-2, 5e-2])
        ax.set_ylim([-5e-2, 5e-2])
        ax.set_zlim([-5e-2, 5e-2])
        ax.set_title(f"Iteration {i + 1}: Reconstructed Object")
        plt.pause(1)