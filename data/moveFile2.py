import os
import shutil
import random
from collections import defaultdict


def move_random_files(source_dir, target_dir, num):
    """
    从源目录中为每个设备随机选择num个文件，移动到目标目录

    参数:
        source_dir: 源文件夹路径
        target_dir: 目标文件夹路径
        num: 每个设备要移动的文件数量
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 按设备编号分组文件
    device_files = defaultdict(list)

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        # 检查文件名是否符合格式 device_xx_yyyy.npy
        if filename.startswith("device_") and filename.endswith(".npy"):
            parts = filename.split("_")
            if len(parts) >= 3:
                device_id = parts[1]  # 获取设备编号xx
                file_path = os.path.join(source_dir, filename)
                device_files[device_id].append(file_path)

    # 对每个设备处理
    total_moved = 0
    for device_id, files in device_files.items():
        # 确保有足够的文件可供选择
        available = len(files)
        if available == 0:
            continue

        # 如果文件数量不足num，则全部移动
        select_count = min(num, available)

        # 随机选择文件
        selected_files = random.sample(files, select_count)

        # 移动文件
        for file_path in selected_files:
            try:
                # 保留原文件名移动到目标目录
                shutil.move(file_path, os.path.join(target_dir, os.path.basename(file_path)))
                total_moved += 1
            except Exception as e:
                print(f"移动文件 {file_path} 时出错: {e}")

        print(f"设备 {device_id}: 从 {available} 个文件中移动了 {select_count} 个文件")

    print(f"操作完成，共移动了 {total_moved} 个文件到 {target_dir}")

# 一个文件夹中有一批 device_xx_yyyy.npy 文件。
# 其中 xx是设备编号，yyyy是这个设备的某个编号文件。
# 这个文件夹有多个设备，每个设备有多个文件 。现在使用python帮我写一个程序，将这个文件夹下的所有设备随机取出num参数个文件移动到特定的目录。
if __name__ == "__main__":
    # 示例配置
    SOURCE_DIRECTORY = "G:/seidataforCIL/train"  # 源文件夹路径
    TARGET_DIRECTORY = "G:/seidataforCIL/test-closed"  # 目标文件夹路径
    NUM_TO_SELECT = 100  # 每个设备要选择的文件数量

    # 执行移动操作
    move_random_files(SOURCE_DIRECTORY, TARGET_DIRECTORY, NUM_TO_SELECT)
