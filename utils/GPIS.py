"""
@FileName：GPIS.py
@Description：
@Author：Ferry
@Time：4/10/25 2:54 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import open3d as o3d
import numpy as np

import gpytorch

import time
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.ion()


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
    def __init__(self, res, display_percentile_low, display_percentile_high, training_iter, grid_count, min_data, max_data):
        self.res = res
        self.display_percentile_low = display_percentile_low
        self.display_percentile_high = display_percentile_high
        self.training_iter = training_iter
        self.grid_size = grid_count
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.min_data = min_data.reshape(3) #x, y, z
        self.max_data = max_data.reshape(3) #x, y, z

        x_axis = np.linspace(self.min_data[0], self.max_data[0], grid_count)
        y_axis = np.linspace(self.min_data[1], self.max_data[1], grid_count)
        z_axis = np.linspace(self.min_data[2], self.max_data[2], grid_count)

        origin_X, origin_y = [], []
        for x in x_axis:
            for y in y_axis:
                for z in z_axis:
                    origin_X.append([x, y, z])
                    origin_y.append(1)

        self.origin_X = np.array(origin_X)
        self.origin_y = np.array(origin_y)

        self.k_nearest = 50


        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter_plot = self.ax.scatter([], [], [], c='blue', marker='o', s=10, alpha=0.5, label='3D Points')
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.set_title('3D Scatter Plot of trainx (incremental)')
        self.ax.legend()
        plt.ion()
        plt.show()

        self.buffer_explored_x = np.empty((0, 3))
        self.buffer_explored_y = np.empty((0))
        self.touched_x = np.empty((0, 3))



    def update(self, explored_new_x, explored_new_y):
        """
        Update the model by the new points hands exploration.
        Args:
            explored_new_x:  The new points axis of hand has explored.
            explored_new_y:  The new points values of hand has explored. (0--untouched 1--touched)

        Returns:
            None.
        """
        self.buffer_explored_x = np.vstack([self.buffer_explored_x, explored_new_x])
        self.buffer_explored_y = np.concatenate([self.buffer_explored_y, explored_new_y])
        self.touched_x = np.vstack([self.touched_x, explored_new_x[np.where(explored_new_y == 1)]])

        gridpcd = o3d.geometry.PointCloud()
        gridpcd.points = o3d.utility.Vector3dVector(self.origin_X)
        self.kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)

        index = []

        for p in self.buffer_explored_x:
            [k, idx, dis_sqr] = self.kdtree_gridpcd.search_knn_vector_3d(p, self.k_nearest)
            for m in range(k):
                index.append(idx[m])

        y_buf = np.zeros_like(self.origin_y)
        mask = np.ones(len(self.origin_X), dtype=bool)
        mask[index] = False
        y_buf[mask] = self.origin_y[mask]

        train_X = np.vstack([self.origin_X, self.buffer_explored_x])
        train_y = np.hstack([y_buf, self.buffer_explored_y])

        self.xstar = np.zeros((self.res ** 3, 3))

        for j in range(self.res):
            for i in range(self.res):
                d = j * self.res ** 2  # Delta
                axis_min = d + self.res * i
                axis_max = self.res * (i + 1) + d

                self.xstar[axis_min:axis_max, 0] = np.linspace(self.min_data[0], self.max_data[0], num=self.res)  # in X
                self.xstar[axis_min:axis_max, 1] = self.min_data[1] + i * ((self.max_data[1] - self.min_data[1]) / self.res)  # in X
                self.xstar[axis_min:axis_max, 2] = self.min_data[2] + ((j + 1) * ((self.max_data[2] - self.min_data[2]) / self.res))

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
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, self.training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
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
            x = torch.FloatTensor(self.buffer_explored_x).cuda()
            observed_pred = self.likelihood(self.model(x))
            original_prediction = observed_pred.mean.cpu().numpy()
            # print(pred.shape)
            self.original_confidence_lower, self.original_confidence_upper = observed_pred.confidence_region()

        self.original_pred_low, self.original_pred_high = np.percentile(original_prediction,
                                                              [self.display_percentile_low, self.display_percentile_high])

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

        threshold = self.original_pred_high

        mask = self.prediction > threshold

        self.estimated_surface = self.xstar[mask]

        N = min(10000, self.estimated_surface.shape[0])
        indices = np.random.choice(self.estimated_surface.shape[0], N, replace=False)
        self.estimated_sampled = self.estimated_surface[indices]

        self.origin_X = np.array(self.estimated_sampled)
        self.origin_y = np.ones((self.estimated_sampled.shape[0],))

        return self.estimated_surface, self.origin_X, self.origin_y


    def draw_pointcloud(self):
        """
        Draw an o3d point cloud on the screen.
        Returns:
            None.
        """

        print("Drawing 3d points cloud")
        print("--------------------------------------------------------")
        fig = go.Figure(go.Scatter3d(
            x=self.estimated_surface[:, 0],
            y=self.estimated_surface[:, 1],
            z=self.estimated_surface[:, 2],
            mode='markers',
            marker=dict(size=2, color='lightblue'),
            name='Estimated Surface Point Cloud'
        ))

        fig.show()

    def draw_3Dpoints(self):
        """
        Draw a plt figure for the sampled 3D points.
        Returns:
            None.
        """
        print("Drawing new origins surface points")
        print("--------------------------------------------------------")
        self.scatter_plot._offsets3d = (
            self.origin_X[:, 0],
            self.origin_X[:, 1],
            self.origin_X[:, 2]
        )

        xlim = [np.min(self.origin_X[:, 0]), np.max(self.origin_X[:, 0])]
        ylim = [np.min(self.origin_X[:, 1]), np.max(self.origin_X[:, 1])]
        zlim = [np.min(self.origin_X[:, 2]), np.max(self.origin_X[:, 2])]
        padding = 0.05
        self.ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
        self.ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
        self.ax.set_zlim(zlim[0] - padding, zlim[1] + padding)
        plt.pause(0.3)

    def draw(self):
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

    def generate_mesh(self, poisson_depth=8, remove_low_density=False):
        if self.estimated_sampled is None:
            return


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.estimated_sampled)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=10)

        self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

        if remove_low_density:
            densities = np.asarray(densities)
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            self.mesh.remove_vertices_by_mask(vertices_to_remove)

        return self.mesh

    def visualize_mesh(self):
        if self.mesh is None:
            print("No mesh provided.")
            return
        self.mesh.paint_uniform_color([0.6, 0.8, 1.0])
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.mesh, axis])