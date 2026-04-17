import open3d as o3d
import torch
import numpy as np
from src.data import Data
from src.visualization import show

# 1. 读取点云
ply_path = "/home/liu/local/superpoint_transformer/data/old/raw/bedroom/bedroom_r.ply"
pcd = o3d.io.read_point_cloud(ply_path)

# 2. 封装为项目要求的 Data 对象
pos = torch.from_numpy(np.asarray(pcd.points)).float()
rgb = torch.from_numpy(np.asarray(pcd.colors)).float()# Open3D 颜色范围是 0-1，需转为 0-255
data = Data(pos=pos, rgb=rgb)

# 3. 启动可视化
show(data, max_points=100000) # 建议限制点数以保证流畅度