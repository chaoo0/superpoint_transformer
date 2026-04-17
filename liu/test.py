import open3d as o3d
import numpy as np
import os
import glob


def verify_and_fix_pointcloud(ply_path, visualize=False):
    print(f"\n=======================================")
    print(f"--- 正在加载 {ply_path} ---")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        print("⚠️ 警告: 这是一个空的点云文件，跳过处理。")
        return

    # 1. 检查物理尺度和坐标极值
    print("\n[1. 坐标信息]")
    print(f"点云数量: {len(points)}")
    print(f"X轴范围: {points[:, 0].min():.2f} 到 {points[:, 0].max():.2f}")
    print(f"Y轴范围: {points[:, 1].min():.2f} 到 {points[:, 1].max():.2f}")
    print(f"Z轴范围: {points[:, 2].min():.2f} 到 {points[:, 2].max():.2f}")

    # # 2. 自动去中心化 (防精度丢失)
    # center = points.mean(axis=0)
    # pcd.translate(-center)
    # print(f"已将点云中心从 {center} 移动至 (0,0,0)")

    # 3. 检查颜色范围
    print("\n[2. 颜色信息]")
    if len(colors) == 0:
        print("⚠️ 警告: 点云不包含颜色信息！S3DIS 模型极其依赖 RGB！")
    else:
        color_min, color_max = colors.min(), colors.max()
        print(f"颜色值类型: {colors.dtype}, 范围: [{color_min}, {color_max}]")
        if color_max <= 1.0:
            print("✅ 颜色范围符合要求 [0, 1]")
        elif color_max > 1.0 and color_max <= 255.0:
            print("⚠️ 颜色范围是 [0, 255]，正在除以 255.0 归一化")
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # 4. 可视化检查坐标轴 (Z轴/蓝色必须向上)
    if visualize:
        print("\n[3. 坐标轴检查]")
        print("👉 请在弹出的窗口中确认：【蓝色箭头】是否指向天花板/重力反方向？")
        # 创建坐标系 (红X, 绿Y, 蓝Z)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, frame])
    else:
        print("\n[3. 坐标轴检查] 已跳过弹窗确认。")

    # 保存修复后的点云
    # fixed_path = ply_path.replace(".ply", "_.ply")
    # o3d.io.write_point_cloud(fixed_path, pcd)
    # print(f"✅ 修复后的点云已保存至: {fixed_path}")


def batch_process_folder(folder_path, visualize_each=False):
    if not os.path.exists(folder_path):
        print(f"❌ 找不到文件夹: {folder_path}")
        return

    # 查找所有的 .ply 文件
    ply_files = glob.glob(os.path.join(folder_path, "*.ply"))

    # 过滤掉已经修复过的文件，避免将 _fixed.ply 再次处理变成 _fixed_fixed.ply
    files_to_process = [f for f in ply_files if not f.endswith("_.ply")]

    if not files_to_process:
        print(f"没有找到需要处理的 .ply 文件 (或都已经处理过了)。")
        return

    print(f"🔍 总共找到 {len(files_to_process)} 个需要处理的 .ply 文件。")

    # 遍历处理每一个文件
    for ply_file in files_to_process:
        verify_and_fix_pointcloud(ply_file, visualize=visualize_each)

    print("\n🎉 所有文件批量修复完成！")


if __name__ == "__main__":
    # 指定您存放 .ply 文件的文件夹路径
    folder_path = "/home/liu/local/superpoint_transformer/data/real/raw"

    # 执行批量处理
    # 由于您之前已经确认过一次坐标系正常，这里默认为 False 以实现全自动静默处理
    # 如果您想再次逐个确认，可以将其改为 visualize_each=True
    batch_process_folder(folder_path, visualize_each=True)