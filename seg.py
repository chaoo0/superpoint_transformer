import os
import sys
import torch
import open3d as o3d
import numpy as np

# 1. 设置项目根目录并添加到路径
PROJECT_ROOT = "/home/liu/local/superpoint_transformer"
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import hydra
from src.utils import init_config
from src.data import Data
from src.datasets.s3dis_config import CLASS_NAMES, CLASS_COLORS


def main():
    # --- 1. 配置文件与路径设置 ---
    ckpt_path = "spt-2_s3dis_fold5.ckpt"

    # ⚠️ 请在这里填入您要测试的 .ply 文件路径
    ply_path = "/home/liu/local/superpoint_transformer/data/old/raw/bedroom/test_fix.pcd"

    # 自动生成输出路径
    base_name = os.path.splitext(ply_path)[0]
    out_ply_path = f"{base_name}_pred.ply"
    out_html_path = f"{base_name}_pred.html"

    print(">>> 初始化配置...")
    cfg = init_config(overrides=[
        "experiment=semantic/s3dis",
        f"ckpt_path={ckpt_path}",
        "datamodule.load_full_res_idx=True"
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. 实例化并加载预训练模型 ---
    print(f">>> 加载模型 (使用 {device})...")
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(cfg.ckpt_path)
    model = model.eval().to(device)
    model.net.store_features = True  # 开启特征存储以支持 HTML 可视化

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    pre_transform = datamodule.pre_transform
    on_device_transform = datamodule.on_device_test_transform

    # --- 3. 读取并处理点云数据 ---
    print(f">>> 读取点云文件: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)

    if len(points_np) == 0:
        print("❌ 错误：点云为空！")
        return

    pos = torch.from_numpy(points_np).float()
    rgb = torch.from_numpy(colors_np).float() if len(colors_np) == len(points_np) else torch.zeros_like(pos)
    y = torch.zeros(pos.shape[0], dtype=torch.long)
    data = Data(pos=pos, rgb=rgb, y=y)

    # --- 4. 预处理与构建分层超点图 (NAG) ---
    print(">>> 正在进行预处理并构建分层超点图 (NAG)...")

    # [安全补丁] 手动计算高度并剔除 GroundElevation，防止找不到地板报错
    data.elevation = pos[:, 2] - pos[:, 2].min()
    pre_transform.transforms = [t for t in pre_transform.transforms if type(t).__name__ != 'GroundElevation']

    nag = pre_transform(data)
    nag_device = on_device_transform(nag.to(device))

    # --- 5. 执行模型推理 ---
    print(">>> 正在执行语义分割推理...")
    with torch.no_grad():
        output = model(nag_device)

    # 计算体素级别语义用于 HTML，计算全分辨率语义用于 PLY
    nag_device[0].semantic_pred = output.voxel_semantic_pred(super_index=nag_device[0].super_index)
    raw_semseg_y = output.full_res_semantic_pred(
        super_index_level0_to_level1=nag_device[0].super_index,
        sub_level0_to_raw=nag_device[0].sub
    )

    # --- 6. 导出结果 ---
    print(">>> 正在保存预测结果...")

    # 6.1 保存预测后的 .ply 文件
    pred_labels = raw_semseg_y.cpu().numpy()
    color_map = np.asarray(CLASS_COLORS)
    # 将预测标签映射为颜色 (注意防越界保护)
    pred_labels_safe = np.clip(pred_labels, 0, len(color_map) - 1)
    pred_colors = color_map[pred_labels_safe] / 255.0

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = pcd.points
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
    o3d.io.write_point_cloud(out_ply_path, pred_pcd)
    print(f"✅ 预测点云已保存至: {out_ply_path}")

    # 6.2 导出 HTML 可视化文件
    nag_device.show(
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        max_points=100000,
        path=out_html_path,
        title="SPT Semantic Segmentation"
    )
    print(f"✅ HTML 网页已保存至: {out_html_path}")


if __name__ == "__main__":
    main()
#
#
#
#
# import os
# import sys
# import torch
# import open3d as o3d
# import numpy as np
#
# # 1. 设置项目根目录并添加到路径
# PROJECT_ROOT = "/home/liu/local/superpoint_transformer"
# sys.path.append(PROJECT_ROOT)
# os.chdir(PROJECT_ROOT)
#
# import hydra
# from src.utils import init_config
# from src.data import Data
#
# from src.datasets.s3dis_config import CLASS_NAMES, CLASS_COLORS
#
# def main():
#     # --- 1. 配置文件与路径设置 ---
#     ckpt_path = "spt-2_s3dis_fold5.ckpt"  # 确保权重文件在这个路径
#     ply_path = "/home/liu/local/superpoint_transformer/data/s3dis/a1/stanford_office_1.ply"  # 您要推理的点云文件
#
#     print(">>> 初始化配置...")
#     # 使用 S3DIS 语义分割任务的配置
#     cfg = init_config(overrides=[
#         "experiment=semantic/s3dis",
#         f"ckpt_path={ckpt_path}",
#         "datamodule.load_full_res_idx=True"  # 开启此项以获取全分辨率预测结果
#     ])
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # --- 2. 实例化并加载预训练模型 ---
#     print(f">>> 加载模型 (使用 {device})...")
#     model = hydra.utils.instantiate(cfg.model)
#     model = model._load_from_checkpoint(cfg.ckpt_path)
#     model = model.eval().to(device)
#
#     # 保留特征用于交互式可视化
#     model.net.store_features = True
#
#     # --- 3. 读取您的自定义点云数据 ---
#     print(f">>> 读取点云文件: {ply_path}")
#     pcd = o3d.io.read_point_cloud(ply_path)
#     pos = torch.from_numpy(np.asarray(pcd.points)).float()
#     rgb = torch.from_numpy(np.asarray(pcd.colors)).float()  # Open3D默认[0,1]范围
#
#     # 填入一个假的语义标签 y 以便顺利通过项目的底层检查机制
#     y = torch.zeros(pos.shape[0], dtype=torch.long)
#     data = Data(pos=pos, rgb=rgb, y=y)
#
#     # --- 4. 动态实例化 Transform 并构建分层分区 (NAG) ---
#     print(">>> 正在进行预处理并构建分层超点图 (NAG)，这可能需要一些时间...")
#
#     # 修复：先实例化 datamodule，然后从中提取已经正确解析和串联好的预处理函数
#     datamodule = hydra.utils.instantiate(cfg.datamodule)
#     pre_transform = datamodule.pre_transform
#     on_device_transform = datamodule.on_device_test_transform
#
#     # # ==========================================
#     # #GroundElevation（地面高程提取）模块寻找地板失败
#     # # 1. 直接用 Z 坐标减去最低点，作为相对高度特征 (Elevation)
#     # data.elevation = pos[:, 2] - pos[:, 2].min()
#     #
#     # # 2. 从预处理管道中强制剔除底层会报错的 GroundElevation 模块
#     # pre_transform.transforms = [
#     #     t for t in pre_transform.transforms
#     #     if type(t).__name__ != 'GroundElevation'
#     # ]
#     # # ==========================================
#
#     # 执行耗时的超点分区计算 (现在它不会再因为找地板而崩溃了)
#     nag = pre_transform(data)
#
#
#     # 将 Data 对象转换为 NAG 对象
#     #nag = pre_transform(data)
#     nag = on_device_transform(nag.to(device))
#
#     # --- 5. 执行模型推理 ---
#     print(">>> 正在执行语义分割推理...")
#     with torch.no_grad():
#         output = model(nag)
#
#     # --- 6. 提取预测结果 ---
#     print(">>> 提取预测结果...")
#     # 计算体素级别 (Level 0) 的语义预测，并附加到 NAG 用于可视化
#     nag[0].semantic_pred = output.voxel_semantic_pred(super_index=nag[0].super_index)
#
#     # 计算原始全分辨率的语义预测标签 (raw_semseg_y 就是每个点的最终类别 0-12)
#     raw_semseg_y = output.full_res_semantic_pred(
#         super_index_level0_to_level1=nag[0].super_index,
#         sub_level0_to_raw=nag[0].sub
#     )
#
#     print(">>> 推理完成！启动可视化工具...")
#     # --- 7. 可视化结果 ---
#     #nag.show(max_points=100000)
#     nag.show(
#         class_names=CLASS_NAMES,
#         class_colors=CLASS_COLORS,
#         max_points=100000
#     )
#
#
#
#
# if __name__ == "__main__":
#     main()