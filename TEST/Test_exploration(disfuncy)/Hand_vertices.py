"""
@FileName：Hand_vertices.py
@Description：
@Author：Ferry
@Time：3/26/25 2:39 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""
import os
import random
from DexGraspNet.grasp_generation.utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh

class Hand_fingers_vertices:
    def __init__(self, grasp_code):
        self.data_path = "data/dataset/dexgraspnet/"
        self.grasp_code = grasp_code[:-4]
        self.mesh_path = "data/meshdata/"

        self.joint_names = [
                                'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
                                'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
                                'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
                                'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
                                'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
                            ]

        self.translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
        self.rot_names = ['WRJRx', 'WRJRy', 'WRJRz']


        use_visual_mesh = False
        hand_file = "mjcf/shadow_hand_vis.xml" if use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"
        self.hand_model = HandModelMJCFLite(
            hand_file,
            "mjcf/meshes")

        self.grasp_data = np.load(self.data_path + self.grasp_code + ".npy", allow_pickle=True)
        self.object_mesh_origin = trimesh.load(os.path.join(self.mesh_path, self.grasp_code, "coacd/decomposed.obj"))

        self.index = 0

        self.hand_vertices = {"robot0:ffdistal_child": None,
                              "robot0:mfdistal_child": None,
                              "robot0:rfdistal_child": None,
                              "robot0:lfdistal_child": None,
                              "robot0:thdistal_child": None}

        self.update(200)

    def get_fingers_vertices(self):
        for link_name in self.hand_vertices.keys():
            self.hand_vertices[link_name] = self.hand_model.mesh[link_name]['vertices']

    def update(self, n):
        qpos = self.grasp_data[self.index]['qpos']
        # qpos = grasp['qpos']
        rot = np.array(transforms3d.euler.euler2mat(
            *[qpos[name] for name in self.rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor([qpos[name] for name in self.translation_names] + rot + [qpos[name]
                                                                                     for name in self.joint_names],
                                 dtype=torch.float, device="cpu").unsqueeze(0)
        self.hand_model.set_parameters(hand_pose)
        self.hand_mesh = self.hand_model.get_trimesh_data(0)
        self.object_mesh = self.object_mesh_origin.copy().apply_scale(self.grasp_data[self.index]["scale"])
        self.index = random.randint(0, len(self.grasp_data) - 1)
        self.get_fingers_vertices()

        # Sample 10 point in every digit
        self.sampled_vertices = {}

        for key, tensor in self.hand_vertices.items():
            indices = torch.randperm(tensor.shape[0])[:n]  # 生成随机排列并取前10
            sampled_tensor = tensor[indices]  # 选取对应的 10 个点
            self.sampled_vertices[key] = sampled_tensor  # 存入新字典

        all_tensors = list(self.sampled_vertices.values())
        self.sampled_tensor = torch.cat(all_tensors, dim=0)
        pass
        # print(combined_tensor)

        # Sample 10 point in every digit
        # all_points = torch.cat(list(self.hand_vertices.values()), dim=0)  # 合并所有点 -> (9610, 3)
        # indices = torch.randperm(all_points.shape[0])[:n]  # 随机选50个索引
        # self.sampled_tensor = all_points[indices]  # 最终形状 (50, 3)


    def visulize_grasp(self):
        (self.hand_mesh + self.object_mesh).show()


if __name__ == '__main__':
    hand = Hand_fingers_vertices(grasp_code="ddg-gd_pan_poisson_017.npy")
    hand.update(20)
    hand.visulize_grasp()
    for link_name in hand.hand_vertices.keys():
        # print(hand.hand_vertices[link_name])
        print(hand.hand_vertices[link_name].shape)

    print(hand.sampled_tensor)