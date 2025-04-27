"""
@FileName：3_Gp_steps_by_steps.py
@Description：
@Author：Ferry
@Time：4/8/25 2:38 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import open3d as o3d
import numpy as np

import gpytorch

import time
import torch
import plotly.graph_objects as go

from torch.utils.data import TensorDataset, DataLoader


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # ThinPlateKernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def pre_process_data(data, k_nearest=5):
    temp_min = np.min(data, axis=0)
    temp_max = np.max(data, axis=0)
    fone = np.ones((data.shape[0],))
    min_data = temp_min - (
            temp_max - temp_min) * 0.2  # We extend the boundaries of the object a bit to evaluate a little bit further
    max_data = temp_max + (temp_max - temp_min) * 0.2

    x_axis = np.linspace(min_data[0], max_data[0], grid_count)
    y_axis = np.linspace(min_data[1], max_data[1], grid_count)
    z_axis = np.linspace(min_data[2], max_data[2], grid_count)

    train_X, train_y = [], []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                train_X.append([x, y, z])
                train_y.append(0)

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    gridpcd = o3d.geometry.PointCloud()
    gridpcd.points = o3d.utility.Vector3dVector(train_X)

    kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)

    index = []

    for p in data:
        [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
        for m in range(k):
            index.append(idx[m])

    mask = np.ones(len(train_X), dtype=bool)
    mask[index] = False
    train_X = train_X[mask]
    train_y = train_y[mask]

    train_X = np.vstack([train_X, data])
    train_y = np.hstack([train_y, fone])

    return train_X, train_y, min_data, max_data


def res_build(res, min_data, max_data):
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

    return xstar, xeva, yeva, zeva


def train(model, train_x, train_y, likelihood, mll, training_iter=200):
    print("Pretraining")
    start = time.time()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    model.train()
    likelihood.train()

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood, mll


def eval(model, data, likelihood):
    print("Evaluating")
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        x = torch.FloatTensor(data).cuda()
        observed_pred = likelihood(model(x))
        original_prediction = observed_pred.mean.cpu().numpy()
        # print(pred.shape)
        original_confidence_lower, original_confidence_upper = observed_pred.confidence_region()

    original_pred_low, original_pred_high = np.percentile(original_prediction,
                                                          [display_percentile_low, display_percentile_high])
    return observed_pred, original_pred_low, original_pred_high


def test(model, xstar, likelihood):
    model.eval()
    likelihood.eval()

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
    return prediction, uncertainty


def draw(prediction, grasp_data, xeva, yeva, zeva, original_pred_low, original_pred_high):
    print("plotting")

    fig = go.Figure(data=
    [
        go.Isosurface(
            x=xeva.flatten(),
            y=yeva.flatten(),
            z=zeva.flatten(),
            value=prediction.flatten(),
            isomin=original_pred_low,  # -np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction),
            isomax=original_pred_high,  # np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction), #right for bunny model
            caps=dict(x_show=False, y_show=False),
            # colorscale='RdBu',
            surface=dict(show=True, count=2, fill=0.6),
        ),
        go.Scatter3d(
            x=grasp_data[:, 0],
            y=grasp_data[:, 1],
            z=grasp_data[:, 2],
            mode='markers',
            marker=dict(color="red")
        ),
    ]
    )
    fig.show()


if __name__ == '__main__':
    grasp_data = np.load("./contact_points_merged.npy")

    res = 100  # 150 # grid resolution # 50

    display_percentile_low = 10
    display_percentile_high = 90
    training_iter = 200
    grid_count = 10

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_plot = ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot of trainx (incremental)')
    ax.legend()
    plt.ion()

    # training by one time
    # train_X, train_y, min_data, max_data = pre_process_data(grasp_data)
    # X = torch.FloatTensor(train_X).cuda()
    # y = torch.FloatTensor(train_y).cuda()
    #
    #
    # xstar, xeva, yeva, zeva = res_build(res, min_data, max_data)
    #
    # likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    # model = ExactGPModel(X, y, likelihood).cuda()
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #
    # model, likelihood, mll = train(model, X, y, likelihood, mll, training_iter)
    #
    # observed_pred, original_pred_low, original_pred_high = eval(model, grasp_data, likelihood)
    #
    # prediction, uncertainty = test(model, xstar, likelihood)
    #
    # draw(prediction, grasp_data, xeva, yeva, zeva, original_pred_low, original_pred_high)


    # training steps by steps
    batch_size = 50
    num_batches = int(np.ceil(len(grasp_data) / batch_size))

    buffer_grasp = np.empty((0, 3))

    for batch_idx in range(num_batches):
        print(f'Batch {batch_idx + 1}/{num_batches}')

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(grasp_data))

        current_batch = grasp_data[start_idx:end_idx]
        buffer_grasp = np.vstack([buffer_grasp, current_batch])

        train_X, train_y, min_data, max_data = pre_process_data(buffer_grasp)
        X = torch.FloatTensor(train_X).cuda()
        y = torch.FloatTensor(train_y).cuda()

        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        model = ExactGPModel(X, y, likelihood).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model, likelihood, mll = train(model, X, y, likelihood, mll, training_iter=200)

        # if batch_idx == 0 or batch_idx == num_batches - 1 or (batch_idx + 1) % 10 == 0:
        #     xstar, xeva, yeva, zeva = res_build(res, min_data, max_data)
        #     observed_pred, original_pred_low, original_pred_high = eval(model, buffer_grasp, likelihood)
        #     prediction, uncertainty = test(model, xstar, likelihood)
            # draw(prediction, buffer_grasp, xevad, yeva, zeva, original_pred_low, original_pred_high)

        xstar, xeva, yeva, zeva = res_build(res, min_data, max_data)
        observed_pred, original_pred_low, original_pred_high = eval(model, buffer_grasp, likelihood)
        prediction, uncertainty = test(model, xstar, likelihood)
        points_sampled = inside_sampled = outsite_sampled = None

        mask = (prediction > original_pred_low) & (prediction < original_pred_high)
        estimated_surface = xstar[mask]

        xlim = [np.min(estimated_surface[:, 0]), np.max(estimated_surface[:, 0])]
        ylim = [np.min(estimated_surface[:, 1]), np.max(estimated_surface[:, 1])]
        zlim = [np.min(estimated_surface[:, 2]), np.max(estimated_surface[:, 2])]
        padding = 0.05
        ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
        ax.set_zlim(zlim[0] - padding, zlim[1] + padding)

        scatter_plot._offsets3d = (
            estimated_surface[:, 0],
            estimated_surface[:, 1],
            estimated_surface[:, 2]
        )
        plt.pause(0.3)
        plt.show()
        print(f"surface low thres:{original_pred_low}")
        print(f"surface high thres:{original_pred_high}")

    threshold = original_pred_high


    mask = prediction > threshold

    estimated_surface = xstar[mask]

    # fig = go.Figure(go.Scatter3d(
    #     x=estimated_surface[:, 0],
    #     y=estimated_surface[:, 1],
    #     z=estimated_surface[:, 2],
    #     mode='markers',
    #     marker=dict(size=2, color='lightblue'),
    #     name='Estimated Surface Point Cloud'
    # ))
    #
    # fig.show()
