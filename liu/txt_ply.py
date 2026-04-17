import numpy as np
import open3d as o3d
import os

# 1. 指定您刚刚解压出来的官方 txt 文件路径
txt_path = "../data/s3dis/Area_5/office_1/office_1.txt"
print(f"正在读取 {txt_path} ...")

# 2. 读取文本数据
data = np.loadtxt(txt_path)

# 3. 拆分坐标和颜色
points = data[:, 0:3]
colors = data[:, 3:6] / 255.0

# 4. 构建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 去中心化
pcd.translate(-points.mean(axis=0))

# 5. 指定您想要的绝对保存路径
output_dir = "/home/liu/local/superpoint_transformer/data/s3dis/a5"
output_filename = "stanford_office_1.ply"

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, output_filename)

# 6. 保存为 ply 文件
o3d.io.write_point_cloud(save_path, pcd)
print(f"✅ 转换成功！已精确保存至: {save_path}")