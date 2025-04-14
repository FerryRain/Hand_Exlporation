"""
@FileName：6_Test_GPIS_class.py
@Description：
@Author：Ferry
@Time：4/11/25 1:43 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
from utils.GPIS import GPIS
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class DataLoader:
    def __init__(self, touched_data, batch_size):
        self.data = touched_data
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(self.data) / self.batch_size))

        self.idx = 0

    def get_data(self):
        if self.idx < self.num_batches:
            start_idx = self.idx * self.batch_size
            end_idx = min((self.idx + 1) * self.batch_size, len(self.data))
            self.idx += 1
            return self.data[start_idx:end_idx]
        else:
            print("All data has extracted successfully")
            return None

if __name__ == '__main__':
    dataloader = DataLoader(np.load("./contact_points_merged.npy"), 50)
    # mindata = np.array([-0.2, -0.4, 0.25])
    # maxdata = np.array([ 0.25, 0,  0.6])
    mindata = np.array([-0.19272414, -0.3701699, 0.20331692])
    maxdata = np.array([0.21254405, -0.05168852, 0.54474298])
    gpis = GPIS(100, 10, 90, 200, 10, mindata, maxdata)

    new_data = dataloader.get_data()
    while new_data is not None:
        gpis.update(new_data, np.ones(new_data.shape[0]))
        estimated_surface, origin_X, origin_y = gpis.predict()
        # gpis.draw_3Dpoints()
        # gpis.draw()
        # gpis.draw_pointcloud()
        new_data = dataloader.get_data()

    mesh = gpis.generate_mesh()
    gpis.visualize_mesh()

