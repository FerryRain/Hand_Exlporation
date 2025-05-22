"""
@FileName：Vis_points.py
@Description：
@Author：Ferry
@Time：2025 4/30/25 7:18 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import open3d as o3d
import numpy as np


points = np.load("../Data/stage8_1/estimated_surface_450.npy")


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd.paint_uniform_color([0.5, 0.5, 0.5])

o3d.visualization.draw_geometries([pcd])
