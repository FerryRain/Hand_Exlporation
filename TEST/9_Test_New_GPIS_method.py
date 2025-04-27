"""
@FileName：9_Test_New_GPIS_method.py
@Description：
@Author：Ferry
@Time：2025 4/24/25 2:47 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import numpy as np

import open3d as o3d
import gpytorch

import time
import torch
import itertools
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

plt.ion()
from mpl_toolkits.mplot3d import Axes3D

# 创建 DataFrame 并可视化
# df = pd.DataFrame(points_valid, columns=["x", "y", "z"])
# df["label"] = labels_valid
#
# fig = px.scatter_3d(df, x='x', y='y', z='z', color=df["label"].astype(str), opacity=0.5)
# fig.update_layout(title="三层球壳结构（三值标签：1/0/-1）")
# fig.show()

res = 100
k_nearest = 5
display_percentile_low = 10
display_percentile_high = 90
training_iter = 500
grid_count = 10


def init_origine(res=10, out=0.3, middle=0.2, inner=0.1, shape='ball'):
    lin = np.linspace(-out, out, res)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    if shape == 'ball':
        distances = np.linalg.norm(points, axis=1)
        labels = np.full(points.shape[0], np.nan)

        labels[(distances >= middle) & (distances < out)] = 1
        labels[(distances >= inner) & (distances < middle)] = 0
        labels[distances < inner] = -1

        valid_mask = ~np.isnan(labels)
        points_valid = points[valid_mask]
        labels_valid = labels[valid_mask]

        return points_valid, labels_valid
    elif shape == 'box':
        labels = np.zeros(points.shape[0])
        inner_mask = np.all(np.abs(points) <= inner / 2, axis=1)
        middle_mask = np.logical_and(
            np.all(np.abs(points) <= middle / 2, axis=1),
            ~inner_mask
        )
        outer_mask = np.logical_and(
            np.all(np.abs(points) <= out / 2, axis=1),
            ~middle_mask & ~inner_mask
        )

        labels[inner_mask] = -1
        labels[middle_mask] = 0
        labels[outer_mask] = 1

        return points, labels


# grasp_data = np.load("./contact_points_merged.npy")
pcd = o3d.io.read_point_cloud("bunny_ascii.pcd")
grasp_data = np.asarray(pcd.points)
temp_min = np.min(grasp_data, axis=0)
temp_max = np.max(grasp_data, axis=0)
fone = np.ones((grasp_data.shape[0],))
min_data = temp_min - (
            temp_max - temp_min) * 0.2  # We extend the boundaries of the object a bit to evaluate a little bit further
max_data = temp_max + (temp_max - temp_min) * 0.2

origin_X, origin_y = init_origine(out=abs(max(max_data)) + abs(min(min_data)), middle=abs(max(max_data)),
                                  inner=abs(min(min_data)))
# df = pd.DataFrame(origin_X, columns=["x", "y", "z"])
# df["label"] = origin_y
#
# fig = px.scatter_3d(df, x='x', y='y', z='z', color=df["label"].astype(str), opacity=0.5)
# fig.update_layout(title="三层球壳结构（三值标签：1/0/-1）")
# fig.show()

batch_size = 5
index = []
num_batches = int(np.ceil(len(grasp_data) / batch_size))

batch_idx = 0
buffer_grasp = np.empty((0, 3))
k_nearest = 5


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # ThinPlateKernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter_plot = ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot of trainx (incremental)')
ax.legend()
plt.ion()
plt.show()
points_sampled = inside_sampled = outsite_sampled= None

for batch_idx in range(num_batches):
    ax.set_title(f'Batch {batch_idx + 1}/{num_batches}')  # 动态更新标题

    # 更新坐标轴范围以居中

    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(grasp_data))
    current_batch = grasp_data[start_idx:end_idx]
    buffer_grasp = np.vstack([buffer_grasp, current_batch])
    batch_idx += 1
    fone = np.zeros((buffer_grasp.shape[0],))
    print("buffer_grasp.shape:", buffer_grasp.shape)

    # for i in range(3):
    #     index = []
    #     gridpcd = o3d.geometry.PointCloud()
    #     gridpcd.points = o3d.utility.Vector3dVector(origin_X[origin_y==])
    #     kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)
    buffer_x = origin_X.copy()
    buffer_y = origin_y.copy()

    if outsite_sampled is not None:
        index = []
        gridpcd = o3d.geometry.PointCloud()
        gridpcd.points = o3d.utility.Vector3dVector(buffer_x)
        kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)
        for p in outsite_sampled:
            [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
            for m in range(k):
                index.append(idx[m])

        index = np.array(index)
        selected_y = origin_y[index]
        index_inside_local = (selected_y == -1)
        index_surface_local = (selected_y == 0)

        index_inside_global = index[index_inside_local]
        index_surface_global = index[index_surface_local]

        buffer_y = origin_y.copy()
        buffer_y[index_inside_global] = 0
        buffer_y[index_surface_global] = 1

    if inside_sampled is not None:
        index = []
        gridpcd = o3d.geometry.PointCloud()
        gridpcd.points = o3d.utility.Vector3dVector(buffer_x)
        kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)
        for p in inside_sampled:
            [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
            for m in range(k):
                index.append(idx[m])

        index = np.array(index)
        selected_y = origin_y[index]
        index_outside_local = (selected_y == 1)
        index_surface_local = (selected_y == 0)

        index_outside_global = index[index_outside_local]
        index_surface_global = index[index_surface_local]

        buffer_y = origin_y.copy()
        buffer_y[index_outside_global] = 0
        buffer_y[index_surface_global] = -1

        # mask = np.ones(len(origin_X), dtype=bool)
        # mask[index_surface_global] = False
        # buffer_x = origin_X[mask]
        # buffer_y = buffer_y[mask]


    index = []
    gridpcd = o3d.geometry.PointCloud()
    gridpcd.points = o3d.utility.Vector3dVector(buffer_x)
    kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)
    if points_sampled is not None:
        for p in np.vstack([points_sampled, buffer_grasp]):
            [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
            for m in range(k):
                index.append(idx[m])
    else:
        for p in buffer_grasp:
            [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
            for m in range(k):
                index.append(idx[m])

    mask = np.ones(len(buffer_x), dtype=bool)
    mask[index] = False
    train_X = buffer_x[mask]
    train_y = buffer_y[mask]

    train_X = np.vstack([train_X, buffer_grasp])
    train_y = np.hstack([train_y, fone])

    xstar = np.zeros((res ** 3, 3))

    for j in range(res):
        for i in range(res):
            d = j * res ** 2  # Delta
            axis_min = d + res * i
            axis_max = res * (i + 1) + d

            xstar[axis_min:axis_max, 0] = np.linspace(min_data[0], max_data[0], num=res)  # in X
            xstar[axis_min:axis_max, 1] = min_data[1] + i * ((max_data[1] - min_data[1]) / res)  # in X
            xstar[axis_min:axis_max, 2] = min_data[2] + ((j + 1) * ((max_data[2] - min_data[2]) / res))

    tsize = res
    xeva = np.reshape(xstar[:, 0], (tsize, tsize, tsize))
    yeva = np.reshape(xstar[:, 1], (tsize, tsize, tsize))
    zeva = np.reshape(xstar[:, 2], (tsize, tsize, tsize))

    X = torch.FloatTensor(train_X).cuda()
    y = torch.FloatTensor(train_y).cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = ExactGPModel(X, y, likelihood).cuda()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    print("Predicting")
    start = time.time()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if points_sampled is not None:
            x = torch.FloatTensor(np.vstack([points_sampled, buffer_grasp])).cuda()
        else:
            x = torch.FloatTensor(buffer_grasp).cuda()
        observed_pred = likelihood(model(x))
        original_prediction = observed_pred.mean.cpu().numpy()
        # print(pred.shape)
        original_confidence_lower, original_confidence_upper = observed_pred.confidence_region()

    original_pred_low, original_pred_high = np.percentile(original_prediction,
                                                          [display_percentile_low, display_percentile_high])
    print(original_pred_low)
    print(original_pred_high)

    _ = time.time()
    prediction_buf = []
    uncertainty_buf = []
    isplit = 0
    step_length = 10000
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        while isplit * step_length < xstar.shape[0]:
            isplit += 1
            split_min = step_length * (isplit - 1)
            split_max = np.minimum(step_length * isplit, xstar.shape[0])

            xstar_tensor = torch.FloatTensor(xstar[split_min:split_max, :]).cuda()
            observed_pred = likelihood(model(xstar_tensor))
            prediction = observed_pred.mean.cpu().numpy()
            prediction_buf.append(prediction)
            confidence_lower, confidence_upper = observed_pred.confidence_region()
            confidence_lower = confidence_lower.cpu().numpy()
            confidence_upper = confidence_upper.cpu().numpy()
            uncertainty_buf.append(confidence_upper - confidence_lower)

    prediction = np.hstack(prediction_buf)
    uncertainty = np.hstack(uncertainty_buf)

    threshold = original_pred_high

    mask = (prediction > original_pred_low) & (prediction < original_pred_high)

    mask_out = prediction > 2 * original_pred_high

    mask_inside = prediction < original_pred_low / 2

    estimated_surface = xstar[mask]
    insite_points = xstar[mask_inside]
    outsite_points = xstar[mask_out]

    N = min(100, estimated_surface.shape[0])
    N_sample = min(1000, estimated_surface.shape[0])
    if len(estimated_surface) != 0:
        indices = np.random.choice(estimated_surface.shape[0], N, replace=False)
        points_sampled = estimated_surface[indices]
        indices_draws = np.random.choice(estimated_surface.shape[0], N_sample, replace=False)
        estimated_surface = estimated_surface[indices_draws]
    else:
        points_sampled = None

    if len(insite_points) != 0:
        indices = np.random.choice(insite_points.shape[0], N, replace=False)
        inside_sampled = insite_points[indices]
    else:
        inside_sampled = None

    if len(outsite_points) != 0:
        indices = np.random.choice(outsite_points.shape[0], N, replace=False)
        outsite_sampled = outsite_points[indices]
    else:
        outsite_sampled = None



    xlim = [np.min(estimated_surface[:, 0]), np.max(estimated_surface[:, 0])]
    ylim = [np.min(estimated_surface[:, 1]), np.max(estimated_surface[:, 1])]
    zlim = [np.min(estimated_surface[:, 2]), np.max(estimated_surface[:, 2])]
    padding = 0.05
    ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
    ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
    ax.set_zlim(zlim[0] - padding, zlim[1] + padding)
    plt.pause(0.3)
    scatter_plot._offsets3d = (
        estimated_surface[:, 0],
        estimated_surface[:, 1],
        estimated_surface[:, 2]
    )
    # origin_X = np.array(points_sampled)
    # origin_y = np.ones((points_sampled.shape[0],))

plt.ioff()  # 关闭交互模式
plt.show()
