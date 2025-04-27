import time

import torch
import gpytorch
import numpy as np
import open3d as o3d


# ==============================
# 1️⃣ 初始化3D立方体区域
# ==============================
def generate_initial_cube():
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    z = np.linspace(-2, 2, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    return np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

def is_surface_point(x, y, z, threshold=0.1):
    return (abs(x) == 2 and abs(y) == 2 or abs(z) == 2)  # 判断是否在立方体边界附近



train_points = torch.tensor(generate_initial_cube(), dtype=torch.float32)
train_values = torch.zeros((train_points.shape[0],), dtype=torch.float32)  # 初始假设是立方体

# for i, point in enumerate(train_points):
#     x, y, z = point
#     if is_surface_point(x, y, z):
#         train_values[i] = 1.0  # 表面区域为 1
#     else:
#         train_values[i] = 0.0  # 非表面区域为 0


# ==============================
# 2️⃣ 定义 GP 预测模型
# ==============================
class ObjectShapeGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ObjectShapeGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ObjectShapeGP(train_points, train_values, likelihood)


# ==============================
# 3️⃣ 物体探索
# ==============================
def explore_and_update_gp(model, likelihood, train_x, train_y):
    """ 仅从轮廓向内探索，并删除无效点 """
    num_samples = 50
    surface_mask = train_y > 0.5  # 选取轮廓点
    surface_mask = np.random.randint(0, len(train_x), size=50)
    surface_points = train_x[surface_mask]

    if len(surface_points) < num_samples:
        num_samples = len(surface_points)
    sample_indices = np.random.choice(len(surface_points), num_samples, replace=False)
    sampled_points = surface_points[sample_indices]

    def object_shape(x, y, z):
        return (x ** 2 + y ** 2 + z ** 2) < 1.5  # 真实物体是球体

    observed_values = []
    for p in sampled_points:
        if object_shape(p[0], p[1], p[2]):
            observed_values.append(1.0)  # 触摸到物体
        else:
            observed_values.append(float('nan'))  # 该点为空气

    observed_values = torch.tensor(observed_values, dtype=torch.float32)

    # 删除外部无效点
    mask_valid = ~torch.isnan(train_y)
    train_x = train_x[mask_valid]
    train_y = train_y[mask_valid]

    mask_observed = ~torch.isnan(observed_values)
    sampled_points = sampled_points[mask_observed]
    observed_values = observed_values[mask_observed]

    train_x = torch.cat([train_x, sampled_points], dim=0)
    train_y = torch.cat([train_y, observed_values], dim=0)

    # 重新训练 GP
    model.set_train_data(train_x, train_y, strict=False)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(50):  # 训练 50 轮
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.nan_to_num(0.0))
        loss.backward()
        optimizer.step()

    return model, train_x, train_y


# ==============================
# 4️⃣ 使用 Open3D 进行点云可视化
# ==============================
# def create_visualizer():
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     return vis
#
# # 实时更新点云
# def update_visualization(vis, pcd, points, colors=None):
#     pcd.points = o3d.utility.Vector3dVector(points)
#     if colors is not None:
#         pcd.colors = o3d.utility.Vector3dVector(colors)
#     vis.add_geometry(pcd)
#     vis.update_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
#
# # 创建可视化窗口
# vis = create_visualizer()
# pcd = o3d.geometry.PointCloud()
# colors = np.zeros_like(train_points)
# mask = (train_values > 0.9) & (~torch.isnan(train_values).numpy())
# train_points = train_points[mask]
# train_values = train_values[mask]
# update_visualization(vis, pcd, train_points.numpy(), colors)
# # train_values
# # def visualize_point_cloud(points, colors=None):
# #     pcd = o3d.geometry.PointCloud()
# #     pcd.points = o3d.utility.Vector3dVector(points)
# #
# #     if colors is not None:
# #         pcd.colors = o3d.utility.Vector3dVector(colors)
# #
# #     o3d.visualization.draw_geometries([pcd])
# #
# #
# # # 展示初始立方体点云
# # visualize_point_cloud(train_points.numpy())
#
#
# for i in range(5):  # 逐步探索 5 轮
#     model, train_points, train_values = explore_and_update_gp(model, likelihood, train_points, train_values)
#
#     model.eval()
#     likelihood.eval()
#
#     with torch.no_grad():
#         pred_dist = model(train_points)
#         mean = pred_dist.mean.numpy()
#
#     # 只保留物体点
#     mask = (mean > 0.9) & (~torch.isnan(train_values).numpy())
#     object_points = train_points.numpy()[mask]
#
#     # train_points = train_points[mask]
#     # train_values = train_values[mask]
#     # 颜色：物体点为蓝色
#     colors = np.zeros_like(object_points)
#     colors[:, 2] = 0
#
#     # colors[:, 2] = torch.where(torch.tensor(mean) > 1.0, 1.0, 0.0)
#
#     # visualizer.update(object_points, colors)
#     # visualize_point_cloud(object_points)
#     update_visualization(vis, pcd, object_points, colors)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(2000):  # 逐步探索 5 轮
    model, train_points, train_values = explore_and_update_gp(model, likelihood, train_points, train_values)

    # 预测整个 3D 立方体
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred_dist = model(train_points)
        mean = pred_dist.mean.numpy()

    # 可视化更新后的物体形状
    ax.clear()
    mask = mean > 0.5  # 预测物体的点
    ax.scatter(train_points[:, 0][mask], train_points[:, 1][mask], train_points[:, 2][mask],
               c=mean[mask], marker='o')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_title(f"Iteration {i+1}: Reconstructed Object")
    plt.pause(1)  # 实时更新
