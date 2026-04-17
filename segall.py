import os
import sys
import glob
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
    ckpt_path = "spt-2_s3dis_fold5.ckpt"  # 确保权重文件在这个路径
    folder_path = "data/real/raw"  # 您存放点云的文件夹

    print(">>> 初始化配置...")
    # 使用 S3DIS 语义分割任务的配置
    cfg = init_config(overrides=[
        "experiment=semantic/s3dis",
        f"ckpt_path={ckpt_path}",
        "datamodule.load_full_res_idx=True"  # 开启此项以获取全分辨率预测结果
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. 实例化并加载预训练模型 (整个批处理只需加载一次) ---
    print(f">>> 加载模型 (使用 {device})...")
    model = hydra.utils.instantiate(cfg.model)
    model = model._load_from_checkpoint(cfg.ckpt_path)
    model = model.eval().to(device)

    # 实例化 datamodule 获取 Transform 管道
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    pre_transform = datamodule.pre_transform
    on_device_transform = datamodule.on_device_test_transform

    # --- 3. 查找所有需要处理的 .ply 文件 ---
    # 优先查找经过修复的点云文件，同时排除掉之前已经预测过的 _pred.ply 文件
    all_ply_files = glob.glob(os.path.join(folder_path, "*.ply"))
    ply_files = [f for f in all_ply_files if not f.endswith("_pred.ply")]

    if not ply_files:
        print("❌ 没有找到需要处理的 .ply 文件。")
        return

    print(f">>> 找到 {len(ply_files)} 个 .ply 文件准备进行批量语义分割。")

    # 将类别颜色转换为 Numpy 数组，方便后续快速映射
    color_map = np.asarray(CLASS_COLORS)

    # --- 4. 遍历处理每个文件 ---
    for i, ply_path in enumerate(ply_files):
        print(f"\n[{i + 1}/{len(ply_files)}] 正在处理: {os.path.basename(ply_path)}")

        # 读取点云
        pcd = o3d.io.read_point_cloud(ply_path)
        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors)

        if len(points_np) == 0:
            print("    ⚠️ 空点云，已跳过。")
            continue

        pos = torch.from_numpy(points_np).float()
        # 兼容性保护：确保颜色维度与坐标维度匹配
        if len(colors_np) == len(points_np):
            rgb = torch.from_numpy(colors_np).float()
        else:
            rgb = torch.zeros_like(pos)

        # 填入假的语义标签 y 以通过预处理
        y = torch.zeros(pos.shape[0], dtype=torch.long)
        data = Data(pos=pos, rgb=rgb, y=y)

        # 预处理与构建 NAG
        print("    -> 正在构建分层超点图 (NAG)...")
        nag = pre_transform(data)
        nag = on_device_transform(nag.to(device))

        # 执行推理
        print("    -> 正在执行语义分割推理...")
        with torch.no_grad():
            output = model(nag)

        # 提取全分辨率预测结果
        print("    -> 正在映射预测标签并保存结果...")
        raw_semseg_y = output.full_res_semantic_pred(
            super_index_level0_to_level1=nag[0].super_index,
            sub_level0_to_raw=nag[0].sub
        )

        # --- 5. 将预测标签转为颜色并保存为新的 .ply 文件 ---
        # raw_semseg_y 是一个包含 0-12 类别索引的 Tensor
        pred_labels = raw_semseg_y.cpu().numpy()

        # 根据预测出的类别数字，直接从色板中取出对应的颜色
        # Open3D 要求颜色在 [0, 1] 之间，因此除以 255.0
        pred_colors = color_map[pred_labels] / 255.0

        # 覆写点云颜色并保存为一个新的文件
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = pcd.points  # 保留原始坐标
        pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)  # 写入语义颜色

        # 生成保存路径 (例如 filename_pred.ply)
        save_path = ply_path.replace(".ply", "_pred.ply")
        o3d.io.write_point_cloud(save_path, pred_pcd)
        print(f"    ✅ 预测结果已成功保存至: {os.path.basename(save_path)}")

    print("\n🎉 所有文件的批量语义分割处理完成！您现在可以使用 MeshLab 查看以 '_pred.ply' 结尾的结果文件。")


if __name__ == "__main__":
    main()