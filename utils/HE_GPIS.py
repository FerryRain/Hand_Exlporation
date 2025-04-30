"""
@FileName：HE_GPIS.py
@Description：
@Author：Ferry
@Time：2025 4/25/25 9:12 AM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import os
from datetime import datetime

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch


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


class GPIS():
    def __init__(self, res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
                 origin_y, min_data, max_data):
        self.res = res
        self.display_percentile_low = display_percentile_low
        self.display_percentile_high = display_percentile_high
        self.training_iter = training_iter
        self.grid_size = grid_count
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.origin_X = origin_X
        self.origin_y = origin_y

        self.buffer_explored_x = np.empty((0, 3))
        self.buffer_explored_y = np.empty((0))
        self.touched_x = np.empty((0, 3))
        self.touched_y = np.empty((0))
        self.untouched_x = np.empty((0, 3))
        self.untouched_y = np.empty((0))

        self.k_nearest = 5
        self.min_data = min_data
        self.max_data = max_data
        self.xstar = np.zeros((self.res ** 3, 3))

        for j in range(self.res):
            for i in range(self.res):
                d = j * self.res ** 2  # Delta
                axis_min = d + self.res * i
                axis_max = self.res * (i + 1) + d

                self.xstar[axis_min:axis_max, 0] = np.linspace(self.min_data[0], self.max_data[0], num=self.res)  # in X
                self.xstar[axis_min:axis_max, 1] = self.min_data[1] + i * (
                        (self.max_data[1] - self.min_data[1]) / self.res)  # in X
                self.xstar[axis_min:axis_max, 2] = self.min_data[2] + (
                        (j + 1) * ((self.max_data[2] - self.min_data[2]) / self.res))

        tsize = self.res
        self.xeva = np.reshape(self.xstar[:, 0], (tsize, tsize, tsize))
        self.yeva = np.reshape(self.xstar[:, 1], (tsize, tsize, tsize))
        self.zeva = np.reshape(self.xstar[:, 2], (tsize, tsize, tsize))

        self.time_step = 0
        # self.init_model()

    def update(self, explored_new_x, explored_new_y):
        """
        Update the model by the new points hands exploration.
        Args:
            explored_new_x:  The new points axis of hand has explored.
            explored_new_y:  The new points values of hand has explored. (0--untouched 1--touched)

        Returns:
            None.
        """
        train_X = self.origin_X.copy()
        train_y = self.origin_y.copy()

        self.buffer_explored_x = np.vstack([self.buffer_explored_x, explored_new_x])
        self.buffer_explored_y = np.hstack([self.buffer_explored_y, explored_new_y])
        self.touched_x = np.vstack([self.touched_x, explored_new_x[np.where(explored_new_y == 1)]])
        self.touched_y = np.zeros_like((len(self.touched_x)))
        self.untouched_x = np.vstack([self.untouched_x, explored_new_x[np.where(explored_new_y == 0)]])
        self.untouched_y = np.ones_like((len(self.untouched_x)))

        gridpcd = o3d.geometry.PointCloud()
        gridpcd.points = o3d.utility.Vector3dVector(self.origin_X)
        self.kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)

        index = []

        for p in self.buffer_explored_x:
            [k, idx, dis_sqr] = self.kdtree_gridpcd.search_knn_vector_3d(p, self.k_nearest)
            for m in range(k):
                index.append(idx[m])

        # for p in self.touched_x:
        #     [k, idx, dis_sqr] = self.kdtree_gridpcd.search_knn_vector_3d(p, self.k_nearest)
        #     for m in range(k):
        #         index.append(idx[m])
        mask = np.ones(len(train_X), dtype=bool)
        mask[index] = False
        train_X = train_X[mask]
        train_y = train_y[mask]

        train_X = np.vstack([train_X, self.buffer_explored_x])
        train_y = np.hstack([train_y, self.buffer_explored_y])

        # gridpcd = o3d.geometry.PointCloud()
        # gridpcd.points = o3d.utility.Vector3dVector(self.origin_X)
        # self.kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)
        # index = []
        # for p in self.untouched_x:
        #     [k, idx, dis_sqr] = self.kdtree_gridpcd.search_knn_vector_3d(p, self.k_nearest)
        #     for m in range(k):
        #         index.append(idx[m])
        #
        # y_buf = np.zeros_like(self.origin_y)
        # mask = np.ones(len(self.origin_X), dtype=bool)
        # mask[index] = False
        # y_buf[mask] = self.origin_y[mask]
        #
        # train_X = np.vstack([self.origin_X, self.untouched_x])
        # train_y = np.hstack([y_buf, np.zeros(self.touched_x.shape[0])])

        self.xstar = np.zeros((self.res ** 3, 3))

        for j in range(self.res):
            for i in range(self.res):
                d = j * self.res ** 2  # Delta
                axis_min = d + self.res * i
                axis_max = self.res * (i + 1) + d

                self.xstar[axis_min:axis_max, 0] = np.linspace(self.min_data[0], self.max_data[0], num=self.res)  # in X
                self.xstar[axis_min:axis_max, 1] = self.min_data[1] + i * (
                        (self.max_data[1] - self.min_data[1]) / self.res)  # in X
                self.xstar[axis_min:axis_max, 2] = self.min_data[2] + (
                        (j + 1) * ((self.max_data[2] - self.min_data[2]) / self.res))

        tsize = self.res
        self.xeva = np.reshape(self.xstar[:, 0], (tsize, tsize, tsize))
        self.yeva = np.reshape(self.xstar[:, 1], (tsize, tsize, tsize))
        self.zeva = np.reshape(self.xstar[:, 2], (tsize, tsize, tsize))

        X = torch.FloatTensor(train_X).cuda()
        y = torch.FloatTensor(train_y).cuda()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = ExactGPModel(X, y, self.likelihood).cuda()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        print("Updating model")
        print("--------------------------------------------------------")

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, self.training_iter, loss.item(),
            #     self.model.covar_module.base_kernel.lengthscale.item(),
            #     self.model.likelihood.noise.item()
            # ))
            optimizer.step()

    def predict(self):
        """
        Predicts the new object surface points.
        Returns:
            self.estimated_surface: The surfaces points of the object model predicted
            self.origin_X: Sampled 3D points.
            self.origin_y: Sampled 3D points.
        """

        print("Predicting model")
        print("--------------------------------------------------------")
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = torch.FloatTensor(self.touched_x).cuda()
            observed_pred = self.likelihood(self.model(x))
            original_prediction = observed_pred.mean.cpu().numpy()
            # print(pred.shape)
            self.original_confidence_lower, self.original_confidence_upper = observed_pred.confidence_region()

        self.original_pred_low, self.original_pred_high = np.percentile(original_prediction,
                                                                        [self.display_percentile_low,
                                                                         self.display_percentile_high])

        prediction_buf = []
        uncertainty_buf = []
        isplit = 0
        step_length = 10000
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # test_x = torch.linspace(0, 1, 51)
            while isplit * step_length < self.xstar.shape[0]:
                isplit += 1
                split_min = step_length * (isplit - 1)
                split_max = np.minimum(step_length * isplit, self.xstar.shape[0])

                xstar_tensor = torch.FloatTensor(self.xstar[split_min:split_max, :]).cuda()
                observed_pred = self.likelihood(self.model(xstar_tensor))
                prediction = observed_pred.mean.cpu().numpy()
                prediction_buf.append(prediction)
                confidence_lower, confidence_upper = observed_pred.confidence_region()
                confidence_lower = confidence_lower.cpu().numpy()
                confidence_upper = confidence_upper.cpu().numpy()
                uncertainty_buf.append(confidence_upper - confidence_lower)

        self.prediction = np.hstack(prediction_buf)
        self.uncertainty = np.hstack(uncertainty_buf)

    def step(self, explored_new_x, explored_new_y):
        self.update(explored_new_x, explored_new_y)
        self.predict()

        mask = (self.prediction > self.original_pred_low) & (self.prediction < self.original_pred_high)
        self.estimated_surface = self.xstar[mask]

        self.time_step += 1

        return self.uncertainty, self.estimated_surface

    def draw_Isosurface(self):
        """
        Plotting the predicting Isosurface.
        Returns:
            None.
        """

        print("plotting the predicting Isosurface.")
        print("--------------------------------------------------------")

        fig = go.Figure(data=
        [
            go.Isosurface(
                x=self.xeva.flatten(),
                y=self.yeva.flatten(),
                z=self.zeva.flatten(),
                value=self.prediction.flatten(),
                isomin=self.original_pred_low,  # -np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction),
                isomax=self.original_pred_high,
                # np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction), #right for bunny model
                caps=dict(x_show=False, y_show=False),
                # colorscale='RdBu',
                surface=dict(show=True, count=2, fill=0.6),
            ),
            go.Scatter3d(
                x=self.touched_x[:, 0],
                y=self.touched_x[:, 1],
                z=self.touched_x[:, 2],
                mode='markers',
                marker=dict(color="red")
            ),
        ]
        )
        fig.show()


class global_HE_GPIS(GPIS):
    def __init__(self, res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
                 origin_y, min_data, max_data, show_points=False, store=False, store_path=None):
        super(global_HE_GPIS, self).__init__(res, display_percentile_low, display_percentile_high, training_iter,
                                             grid_count, origin_X,
                                             origin_y, min_data, max_data)

        self.show_points = show_points
        if self.show_points:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.scatter_plot = self.ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')
            self.ax.set_xlabel('X Axis')
            self.ax.set_ylabel('Y Axis')
            self.ax.set_zlabel('Z Axis')
            self.ax.set_title(f'3D Scatter Plot of trainx step {self.time_step}')
            self.ax.legend()
            plt.ion()
            plt.show()

            plt.pause(0.5)

        self.store = store
        self.store_stem = 100
        if self.store:
            if store_path is None:
                self.store_path = "./data"
            else:
                self.store_path = store_path
            timestamp = datetime.now().strftime("%m_%d_%H_%M")
            self.store_path = self.store_path + timestamp
            os.makedirs(self.store_path, exist_ok=True)

    def step(self, explored_new_x, explored_new_y):
        self.update(explored_new_x, explored_new_y)
        self.predict()

        mask = (self.prediction > self.original_pred_low) & (self.prediction < self.original_pred_high)
        self.estimated_surface = self.xstar[mask]
        self.estimated_surface_uncertainty = self.uncertainty[mask]

        if len(self.estimated_surface) != 0:
            N = min(1000, self.estimated_surface.shape[0])
            indices = np.random.choice(self.estimated_surface.shape[0], N, replace=False)
            draw_points = self.estimated_surface[indices]
            if self.show_points:
                self.draw_pointcloud(draw_points)

        self.time_step += 1
        if self.time_step % self.store_stem == 0 and self.store and len(
                self.estimated_surface) != 0 and self.estimated_surface is not None:
            save_path = os.path.join(self.store_path, f"estimated_surface_{self.time_step}.npy")
            np.save(save_path, self.estimated_surface)
            print("--------------------------------------------------------")
            print(f"saved estimated surface estimated_surface_{self.time_step}.npy to", save_path)

        return self.uncertainty, self.xstar, self.estimated_surface, self.estimated_surface_uncertainty, self.prediction

    def draw_pointcloud(self, points):
        """
        Draw a plt figure for the sampled 3D points.
        Returns:
            None.
        """
        print("Drawing new origins surface points")
        print("--------------------------------------------------------")
        self.scatter_plot._offsets3d = (
            points[:, 0],
            points[:, 1],
            points[:, 2]
        )
        if len(points) > 0:
            xlim = [np.min(points[:, 0]), np.max(points[:, 0])]
            ylim = [np.min(points[:, 1]), np.max(points[:, 1])]
            zlim = [np.min(points[:, 2]), np.max(points[:, 2])]
            padding = 0.05
            self.ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
            self.ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
            self.ax.set_zlim(zlim[0] - padding, zlim[1] + padding)
            self.ax.set_title(f'3D Scatter Plot of trainx step {self.time_step}')
        plt.pause(0.5)


class local_HE_GPIS(GPIS):
    def __init__(self, res, display_percentile_low, display_percentile_high, training_iter, grid_count, origin_X,
                 origin_y, min_data, max_data, show_points=False, slide_surface=False, surface_low=-0.02,
                 surface_high=0.02):
        super(local_HE_GPIS, self).__init__(res, display_percentile_low, display_percentile_high, training_iter,
                                            grid_count, origin_X,
                                            origin_y, min_data, max_data)
        self.show_points = show_points
        if self.show_points:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.scatter_plot = self.ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')
            self.ax.set_xlabel('X Axis')
            self.ax.set_ylabel('Y Axis')
            self.ax.set_zlabel('Z Axis')
            self.ax.set_title(f'3D Scatter Plot of trainx step {self.time_step}')
            self.ax.legend()
            plt.ion()
            plt.show()

            plt.pause(0.5)

        self.slide_surface = slide_surface
        self.surface_low = surface_low
        self.surface_high = surface_high

    def step(self, explored_new_x, explored_new_y):
        self.update(explored_new_x, explored_new_y)
        self.predict()

        if self.slide_surface:
            mask = (self.prediction > self.original_pred_low) & (self.prediction < self.original_pred_high)
        else:
            mask = (self.prediction > self.surface_low) & (self.prediction < self.surface_high)
        self.estimated_surface = self.xstar[mask]

        if len(self.estimated_surface) != 0:
            N = min(1000, self.estimated_surface.shape[0])
            indices = np.random.choice(self.estimated_surface.shape[0], N, replace=False)
            draw_points = self.estimated_surface[indices]
            if self.show_points:
                self.draw_pointcloud(draw_points)

        return self.uncertainty, self.estimated_surface

    def draw_pointcloud(self, points):
        """
        Draw a plt figure for the sampled 3D points.
        Returns:
            None.
        """
        print("Drawing new origins surface points")
        print("--------------------------------------------------------")
        self.scatter_plot._offsets3d = (
            points[:, 0],
            points[:, 1],
            points[:, 2]
        )

        xlim = [np.min(points[:, 0]), np.max(points[:, 0])]
        ylim = [np.min(points[:, 1]), np.max(points[:, 1])]
        zlim = [np.min(points[:, 2]), np.max(points[:, 2])]
        padding = 0.05
        self.ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        self.ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
        self.ax.set_zlim(zlim[0] - padding, zlim[1] + padding)
        self.ax.set_title(f'3D Scatter Plot of trainx step {self.time_step}')
        plt.pause(0.5)


class HE_GPIS():
    def __init__(self, res, display_percentile_low, display_percentile_high, training_iter, grid_count, min_data,
                 max_data, show_points=False, slide_surface=False, surface_low=-0.02, surface_high=0.02):
        self.min_data = min_data.reshape(3)  # x, y, z
        self.max_data = max_data.reshape(3)  # x, y, z

        x_axis = np.linspace(self.min_data[0], self.max_data[0], grid_count)
        y_axis = np.linspace(self.min_data[1], self.max_data[1], grid_count)
        z_axis = np.linspace(self.min_data[2], self.max_data[2], grid_count)

        x_bins = np.linspace(min_data[0], max_data[0], 4)
        y_bins = np.linspace(min_data[1], max_data[1], 4)
        z_bins = np.linspace(min_data[2], max_data[2], 4)

        subregion_X = [[] for _ in range(27)]
        subregion_y = [[] for _ in range(27)]
        self.subregion_bounds = [None for _ in range(27)]

        self.origin_X = np.array([[x, y, z] for x in x_axis for y in y_axis for z in z_axis])
        self.origin_y = np.ones((len(self.origin_X),))

        for xi in range(3):
            for yi in range(3):
                for zi in range(3):
                    idx = xi * 9 + yi * 3 + zi
                    sub_min = np.array([x_bins[xi], y_bins[yi], z_bins[zi]])
                    sub_max = np.array([x_bins[xi + 1], y_bins[yi + 1], z_bins[zi + 1]])
                    self.subregion_bounds[idx] = (sub_min, sub_max)

        for i in range(len(self.origin_X)):
            point = self.origin_X[i]
            label = self.origin_X[i]
            x, y, z = point

            xi = np.searchsorted(x_bins, x, side='right') - 1
            yi = np.searchsorted(y_bins, y, side='right') - 1
            zi = np.searchsorted(z_bins, z, side='right') - 1

            xi = min(max(xi, 0), 2)
            yi = min(max(yi, 0), 2)
            zi = min(max(zi, 0), 2)

            region_idx = xi * 9 + yi * 3 + zi
            subregion_X[region_idx].append(point)
            subregion_y[region_idx].append(label)

        # subregion_X[i]: 第i个子区域的所有点坐标。
        self.subregion_X = [np.array(block) for block in subregion_X]
        self.subregion_y = [np.array(block) for block in subregion_y]

        self.global_GPIS = global_HE_GPIS(res, display_percentile_low, display_percentile_high, training_iter,
                                          grid_count, self.origin_X, self.origin_y, self.min_data, self.max_data, )
        self.local_GPIS = [
            local_HE_GPIS(res, display_percentile_low, display_percentile_high, training_iter, grid_count,
                          self.subregion_X[i], self.subregion_y[i], self.subregion_bounds[i, 0],
                          self.subregion_bounds[i, 1]) for i in range(len(self.subregion_X))]


if __name__ == '__main__':
    grasp_data = np.load("../TEST/contact_points_merged.npy")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter_plot = ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot of trainx (incremental)')
    ax.legend()
    plt.ion()

    res = 100  # 150 # grid resolution # 50

    display_percentile_low = 10
    display_percentile_high = 90
    training_iter = 200
    grid_count = 10

    batch_size = 50
    num_batches = int(np.ceil(len(grasp_data) / batch_size))

    buffer_grasp = np.empty((0, 3))

    temp_min = np.min(grasp_data, axis=0)
    temp_max = np.max(grasp_data, axis=0)
    fone = np.ones((grasp_data.shape[0],))
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
                train_y.append(1)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    gpis = GPIS(res, display_percentile_low, display_percentile_high, training_iter, grid_count, train_X, train_y,
                min_data, max_data)

    for batch_idx in range(num_batches):
        print(f'Batch {batch_idx + 1}/{num_batches}')

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(grasp_data))

        current_batch = grasp_data[start_idx:end_idx]
        current_y = np.zeros((current_batch.shape[0],))
        buffer_grasp = np.vstack([buffer_grasp, current_batch])
        uncertainty, estimated_surface = gpis.step(current_batch, current_y)

        print("Drawing new origins surface points")
        print("--------------------------------------------------------")
        N = min(1000, estimated_surface.shape[0])
        indices = np.random.choice(estimated_surface.shape[0], N, replace=False)
        scatter_plot._offsets3d = (
            estimated_surface[indices, 0],
            estimated_surface[indices, 1],
            estimated_surface[indices, 2]
        )

        xlim = [np.min(estimated_surface[:, 0]), np.max(estimated_surface[:, 0])]
        ylim = [np.min(estimated_surface[:, 1]), np.max(estimated_surface[:, 1])]
        zlim = [np.min(estimated_surface[:, 2]), np.max(estimated_surface[:, 2])]
        padding = 0.05
        ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
        ax.set_zlim(zlim[0] - padding, zlim[1] + padding)
        plt.pause(0.5)
