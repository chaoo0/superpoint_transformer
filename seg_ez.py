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
    # 分别指定主模型和分区模型的权重文件
    semantic_ckpt = "ezsp_semantic_s3dis_fold5.ckpt"
    partition_ckpt = "ezsp_partition_s3dis_fold5.ckpt"

    # 定义三大文件夹路径
    RAW_DIR = "data/real"  # 存放原始 .ply
    NAG_DIR = "data/real/nag_ez"  # 存放计算好的超点图 .pt 文件
    HTML_DIR = "data/real/html_ez"  # 存放最终的 HTML 可视化网页

    # 自动创建输出文件夹
    os.makedirs(NAG_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)

    print(">>> 初始化配置...")
    # 核心修改：切换到 EZ-SP 实验配置，并同时传入两个权重路径
    cfg = init_config(overrides=[
        "experiment=semantic/s3dis_ezsp",  # 切换为 ezsp 的专属配置
        f"ckpt_path={semantic_ckpt}",  # 加载语义主模型
        f"datamodule.pretrained_cnn_ckpt_path={partition_ckpt}",  # 供数据模块划分使用
        f"model.pretrained_cnn_ckpt_path={partition_ckpt}",  # 供主模型前端特征提取使用 (解决报错的关键)
        "datamodule.load_full_res_idx=True"
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. 实例化并加载预训练模型 ---
    print(f">>> 加载模型 (使用 {device})...")
    model = hydra.utils.instantiate(cfg.model)

    # 核心修改：通过 kwargs 强制传入本地的分区模型路径，覆盖掉旧 checkpoint 里失效的路径
    model = model._load_from_checkpoint(
        cfg.ckpt_path,
        pretrained_cnn_ckpt_path=partition_ckpt
    )

    model = model.eval().to(device)
    model.net.store_features = True  # 开启特征存储以支持丰富的 HTML 可视化

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    pre_transform = datamodule.pre_transform
    on_device_transform = datamodule.on_device_test_transform

    # --- 3. 查找所有需要处理的 .ply 文件 ---
    ply_files = glob.glob(os.path.join(RAW_DIR, "*.ply"))

    if not ply_files:
        print(f"❌ 在 {RAW_DIR} 中没有找到 .ply 文件。")
        return

    print(f">>> 找到 {len(ply_files)} 个 .ply 文件，开始使用 EZ-SP 批量处理。")

    # --- 4. 遍历处理每个文件 ---
    for i, ply_path in enumerate(ply_files):
        filename = os.path.basename(ply_path)
        base_name = filename.replace(".ply", "")

        nag_path = os.path.join(NAG_DIR, f"{base_name}.pt")
        html_path = os.path.join(HTML_DIR, f"{base_name}.html")

        print(f"\n[{i + 1}/{len(ply_files)}] 正在处理: {filename}")

        # ==========================================
        # 步骤 A：获取超点图 (NAG) - 优先从硬盘读取缓存
        # ==========================================
        if os.path.exists(nag_path):
            print("    -> [跳过计算] 发现已保存的 NAG 缓存，直接加载...")
            nag = torch.load(nag_path)
        else:
            print("    -> [计算阶段] 正在读取点云并构建分层超点图 (NAG)...")
            pcd = o3d.io.read_point_cloud(ply_path)
            points_np = np.asarray(pcd.points)
            colors_np = np.asarray(pcd.colors)

            if len(points_np) == 0:
                print("    ⚠️ 空点云，已跳过。")
                continue

            # 安全降采样保护 (防止之前室外街道数据导致显存溢出)
            if len(points_np) > 500000:
                print(f"    ⚠️ 点云过大 ({len(points_np)} 点)，正在安全降采样以防显存崩溃...")
                pcd = pcd.voxel_down_sample(voxel_size=0.08)
                points_np = np.asarray(pcd.points)
                colors_np = np.asarray(pcd.colors)

            pos = torch.from_numpy(points_np).float()
            rgb = torch.from_numpy(colors_np).float() if len(colors_np) == len(points_np) else torch.zeros_like(pos)
            y = torch.zeros(pos.shape[0], dtype=torch.long)
            data = Data(pos=pos, rgb=rgb, y=y)

            # 执行耗时的超点分区计算 (EZ-SP 架构下此阶段 CPU 主要做轻量预处理)
            nag = pre_transform(data)

            # 将生成的 NAG 保存到专属文件夹
            torch.save(nag, nag_path)
            print(f"    -> [存储阶段] NAG 已成功保存至: {nag_path}")

        # ==========================================
        # 步骤 B：执行模型推理
        # ==========================================
        print("    -> [推理阶段] 正在向 GPU 传输图数据并执行语义分割...")
        nag_device = on_device_transform(nag.to(device))

        with torch.no_grad():
            output = model(nag_device)

        # 提取体素级别的语义预测，并附加到 nag 对象中，这是 HTML 可视化界面的要求
        nag_device[0].semantic_pred = output.voxel_semantic_pred(super_index=nag_device[0].super_index)

        # ==========================================
        # 步骤 C：导出为交互式 HTML
        # ==========================================
        print(f"    -> [导出阶段] 正在生成并保存 HTML 可视化网页...")
        # 调用原生 show 函数，通过 path 参数直接将交互式界面写成文件
        nag_device.show(
            class_names=CLASS_NAMES,
            class_colors=CLASS_COLORS,
            max_points=100000,
            path=html_path,
            title=f"EZ-SP Segmentation - {base_name}"  # 网页标题
        )
        print(f"    ✅ 成功导出: {html_path}")

    print(f"\n🎉 恭喜！所有文件的处理均已完成。")
    print(f"📁 预处理缓存已存在: {NAG_DIR}")
    print(f"🌐 HTML网页结果请查看: {HTML_DIR}")


if __name__ == "__main__":
    main()