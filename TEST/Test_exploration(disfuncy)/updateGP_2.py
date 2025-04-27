import torch
import gpytorch
import open3d as o3d
import numpy as np

# 配置参数
L = 5  # 立方体边长
N = 10 # 每个维度采样点数
delta = 0.1  # 探索步长
threshold = 0.5  # 存在阈值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 球体参数
sphere_center = torch.tensor([L / 2, L / 2, L / 2], device=device)  # 中心坐标
sphere_radius = 1  # 球体半径


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# 生成网格点
x = torch.linspace(0, L, N, device=device)
y = torch.linspace(0, L, N, device=device)
z = torch.linspace(0, L, N, device=device)
X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
grid_points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)

# 初始化训练数据：立方体表面（假设表面存在物体）
surface_mask = ((X == 0) | (X == L - 1) |
                (Y == 0) | (Y == L - 1) |
                (Z == 0) | (Z == L - 1))
X_train = grid_points[surface_mask.reshape(-1)]
y_train = torch.ones(X_train.shape[0], device=device)

# 初始化模型
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPModel(X_train, y_train, likelihood).to(device)
model.train()
likelihood.train()

# 配置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# 初始训练（减少迭代次数以加快演示）
for i in range(30):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Open3D可视化初始化
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)
pcd = o3d.geometry.PointCloud()
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
vis.add_geometry(pcd)
vis.add_geometry(coord_frame)


def update_visualization(points):
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    colors = np.tile([0.2, 0.5, 1.0], (len(points), 1))  # 浅蓝色表示预测
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()



# 初始预测显示立方体
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(grid_points))
    occupied = pred.mean >= threshold
update_visualization(grid_points[occupied])


# 多轴探索函数
def explore_axis(axis=0):
    current_max = L
    while current_max - delta > 0:
        # 生成采样平面
        sample_plane = current_max - delta
        plane_mask = (grid_points[:, axis] >= sample_plane) & (grid_points[:, axis] < current_max)
        sample_points = grid_points[plane_mask]

        # 计算到球心的距离并生成标签
        dists = torch.norm(sample_points - sphere_center, dim=1)
        y_sample = torch.where(dists <= sphere_radius, 1.0, 0.0)

        # 更新训练数据
        global X_train, y_train
        X_train = torch.cat([X_train, sample_points])
        y_train = torch.cat([y_train, y_sample])

        # 如果区域无物体，标记外侧为空
        if torch.all(y_sample == 0):
            outer_mask = grid_points[:, axis] >= sample_plane
            X_train = torch.cat([X_train, grid_points[outer_mask]])
            y_train = torch.cat([y_train, torch.zeros(outer_mask.sum(), device=device)])
            current_max = sample_plane
        else:
            current_max = sample_plane

        # 重新训练模型（快速微调）
        model.set_train_data(X_train, y_train, strict=False)
        model.train()
        for i in range(50):  # 减少迭代次数
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        # 更新预测
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(grid_points))
            occupied = pred.mean >= threshold
        update_visualization(grid_points[occupied])


# 执行多轴探索
for axis in [0, 1, 2]:  # 沿x,y,z三个轴依次探索
    explore_axis(axis)

# 最终保留窗口
vis.run()
vis.destroy_window()