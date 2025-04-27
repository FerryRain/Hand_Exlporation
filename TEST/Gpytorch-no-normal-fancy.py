## Description
# This Script loads a 3D point cloud in pcd format with its normals. 
# Then performs a 3D Reconstruction using GPIS. 

## Add libraries
import open3d as o3d
import numpy as np

import gpytorch

import time
import torch
import itertools
import plotly.graph_objects as go

def init_origine(res=10, out=0.3, middle=0.2, inner=0.1, shape='ball'):
    lin = np.linspace(-out, out, res)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    if shape == 'ball':
        distances = np.linalg.norm(points, axis=1)
        labels = np.full(points.shape[0], np.nan)

        labels[(distances >= middle) & (distances < out)] = 1
        labels[(distances >= inner) & (distances < middle)] = 0
        labels[distances < inner] = 1

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



# Grid resolution
res = 100 # 150 # grid resolution # 50
k_nearest=5
display_percentile_low=10
display_percentile_high=90
training_iter = 1000
grid_count = 10
pcd = o3d.io.read_point_cloud("bunny_ascii.pcd")
# pcd = o3d.io.read_point_cloud("bowlA.pcd")


original_points = np.asarray(pcd.points)
# v_normals = np.asarray(pcd.normals)
temp_min = np.min(original_points, axis=0)
temp_max = np.max(original_points, axis=0)
fone   = np.zeros((original_points.shape[0],))
min_data = temp_min - (temp_max-temp_min)*0.2 # We extend the boundaries of the object a bit to evaluate a little bit further 
max_data = temp_max + (temp_max-temp_min)*0.2 # the 0.6 value can be adjusted dependeing the size of the bounding box, and if for example you are interested in regions outside the boundaries of the object modelled by the sensors.

kdtree = o3d.geometry.KDTreeFlann(pcd)

x_axis = np.linspace(min_data[0], max_data[0],grid_count)
y_axis = np.linspace(min_data[1], max_data[1],grid_count)
z_axis = np.linspace(min_data[2], max_data[2],grid_count)

train_X, train_y = init_origine(out=abs(max(max_data)) + abs(min(min_data)), middle=abs(max(max_data)),
                                  inner=abs(min(min_data)))

gridpcd = o3d.geometry.PointCloud()
gridpcd.points = o3d.utility.Vector3dVector(train_X)

kdtree_gridpcd = o3d.geometry.KDTreeFlann(gridpcd)

index = []

for p in original_points:
    [k, idx, dis_sqr] = kdtree_gridpcd.search_knn_vector_3d(p, k_nearest)
    for m in range(k):
        index.append(idx[m])

# create a mask for deleting elements
mask = np.ones(len(train_X), dtype=bool)
mask[index] = False
train_X = train_X[mask]
train_y = train_y[mask]

train_X = np.vstack([train_X, original_points])
train_y = np.hstack([train_y, fone])



# purely for display purpose
xstar = np.zeros((res**3, 3))

for j in range(res):
	for i in range(res):
		d = j * res**2 # Delta
		axis_min = d + res * i
		axis_max = res * (i + 1) + d

		xstar[axis_min:axis_max, 0] = np.linspace(min_data[0], max_data[0], num=res) # in X
		xstar[axis_min:axis_max, 1] = min_data[1] + i * ((max_data[1] - min_data[1]) / res) # in X
		xstar[axis_min:axis_max, 2] = min_data[2] + ((j + 1) * ((max_data[2] - min_data[2]) / res))

tsize = res
xeva = np.reshape(xstar[:, 0], (tsize, tsize, tsize))
yeva = np.reshape(xstar[:, 1], (tsize, tsize, tsize))
zeva = np.reshape(xstar[:, 2], (tsize, tsize, tsize))

# print("Expected ((100, 100, 100), (100, 100, 100), (100, 100, 100))")
# print(xeva.shape, yeva.shape, zeva.shape)

# GP Setup
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

# ker = GPy.kern.Exponential(3)
# ker=GPy.kern.RatQuad(3,power=0.8)
X = torch.FloatTensor(train_X).cuda()
y = torch.FloatTensor(train_y).cuda()


likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model =ExactGPModel(X,y,likelihood).cuda()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

model.train()
likelihood.train()

# Query GP
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

#xstar.shape:[res**3,3]
#xstar = torch.FloatTensor(xstar).cuda()

model.eval()
likelihood.eval()

# for display, see the range of fitted original_points

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    x = torch.FloatTensor(original_points).cuda()
    observed_pred = likelihood(model(x))
    original_prediction = observed_pred.mean.cpu().numpy()
    # print(pred.shape)
    original_confidence_lower, original_confidence_upper = observed_pred.confidence_region()

original_pred_low, original_pred_high = np.percentile(original_prediction, [display_percentile_low, display_percentile_high])


_ = time.time()
prediction_buf = []
uncertainty_buf = []
isplit=0
step_length = 10000
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    while isplit*step_length<xstar.shape[0]:
        isplit += 1
        split_min = step_length * (isplit-1)
        split_max = np.minimum(step_length * isplit, xstar.shape[0])

        xstar_tensor = torch.FloatTensor(xstar[split_min:split_max,:]).cuda()
        observed_pred = likelihood(model(xstar_tensor))
        prediction = observed_pred.mean.cpu().numpy()
        prediction_buf.append(prediction)
        confidence_lower, confidence_upper = observed_pred.confidence_region()
        confidence_lower = confidence_lower.cpu().numpy()
        confidence_upper = confidence_upper.cpu().numpy()
        uncertainty_buf.append(confidence_upper - confidence_lower)

prediction = np.hstack(prediction_buf)
uncertainty = np.hstack(uncertainty_buf)
# print(time.time()-_)
# print("grid loop")
output_grid = np.zeros((res, res, res))
for counter, (i, j, k) in enumerate(itertools.product(range(res), range(res), range(res))):
    output_grid[i][j][k] = prediction[counter]


print(f"surface low thres:{original_pred_low}")
print(f"surface high thres:{original_pred_high}")
# x_star_std = np.std(prediction)

print("plotting")


fig = go.Figure(data=
            [go.Isosurface(
                x=xeva.flatten(),
                y=yeva.flatten(),
                z=zeva.flatten(),
                value=prediction.flatten(),
                isomin=original_pred_low,#-np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction),
                isomax=original_pred_high,#np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction), #right for bunny model
                caps=dict(x_show=False, y_show=False),
                # colorscale='RdBu',
                surface= dict(show=True,count=2, fill=0.6),

            ),
            go.Scatter3d(
                x=original_points[:,0],
                y=original_points[:,1],
                z=original_points[:,2],
                mode='markers',
                marker=dict(color="red")
            ),
            # go.Scatter3d(
            #     x=points_in[:,0],
            #     y=points_in[:,1],
            #     z=points_in[:,2],
            #     mode='markers',
            #     marker=dict(color="green")
            # ),
            # go.Scatter3d(
            #     x=points_out[:,0],
            #     y=points_out[:,1],
            #     z=points_out[:,2],
            #     mode='markers',
            #     marker=dict(color="red")
            # )
            ]
    )
fig.show()