import open3d as o3d
import os

def convert_pcd_to_ply(input_folder, output_folder):
    """
    将文件夹内的所有 .pcd 文件转换为 .ply 格式
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    found_files = False
    for filename in os.listdir(input_folder):
        if filename.endswith(".pcd"):
            found_files = True
            pcd_path = os.path.join(input_folder, filename)
            # Open3D 自动处理二进制和压缩的 RGB 字段
            pcd = o3d.io.read_point_cloud(pcd_path)
            
            # 转换为 .ply 格式
            ply_path = os.path.join(output_folder, filename.replace(".pcd", ".ply"))
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"转换成功: {filename} -> {os.path.basename(ply_path)}")
    
    if not found_files:
        print("未在文件夹中找到 .pcd 文件")

# --- 修改这里 ---
if __name__ == "__main__":
    # "." 代表当前脚本所在的文件夹
    convert_pcd_to_ply(".", ".")
