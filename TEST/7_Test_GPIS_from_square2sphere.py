"""
@FileName：7_Test_GPIS_from_square2sphere.py
@Description：
@Author：Ferry
@Time：4/15/25 2:25 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.GPIS import GPIS
import matplotlib.pyplot as plt


import numpy as np


def create_init_surface_points():
    resolution = 20
    x = y = z = np.linspace(-0.5, 0.5, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    abs_X = np.abs(X)
    abs_Y = np.abs(Y)
    abs_Z = np.abs(Z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    half_small = 0.6 / 2

    mask = (
        (abs_X <= half_small) &
        (abs_Y <= half_small) &
        (abs_Z <= half_small)
    ).astype(np.uint8)

    labels = np.zeros(points.shape[0], dtype=np.uint8)
    labels[mask.ravel() == 1] = 1
    return points, labels




if __name__ == '__main__':
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.3
    mindata = np.array([-0.5, -0.5, -0.5])
    maxdata = np.array([0.5, 0.5, 0.5])
    points, labels = create_init_surface_points()
    gpis = GPIS(res=50, display_percentile_low=10, display_percentile_high=90,
                training_iter=200, grid_count=10, min_data=mindata, max_data=maxdata)


    surface_points = gpis.init_model(points, labels)


    for i in range(50):
        surface_points = surface_points
        sample_count = 100
        indices = np.random.choice(surface_points.shape[0], size=sample_count, replace=False)
        sampled_points = surface_points[indices]
        sample_distances = np.linalg.norm(sampled_points - center, axis=1)
        sample_labels = (sample_distances <= 0.2).astype(np.uint8)
        gpis.update(sampled_points, sample_labels)
        estimated_surface_new, _, _ = gpis.predict()
        gpis.draw_3Dpoints()
    gpis.draw_pointcloud()

