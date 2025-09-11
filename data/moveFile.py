import os
import shutil
import random


def move_random_files(source_folder, destination_folder, num_files_to_move):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # 确保要移动的文件数不超过可用文件数
    num_files_to_move = min(num_files_to_move, len(all_files))

    # 随机选择指定数量的文件
    selected_files = random.sample(all_files, num_files_to_move)

    # 移动文件
    for file in selected_files:
        src_path = os.path.join(source_folder, file)
        dest_path = os.path.join(destination_folder, file)
        shutil.move(src_path, dest_path)
        print(f"Moved: {file} -> {destination_folder}")

# modes = ["Q", "Q_ABS", "Q_ABS_FFT", "I_ABS_FFT", "IQ_ABS_FFT"]
# for m in modes:
#     # 示例用法
#     source_folder = f"G:/seidata/32ft-input/{m}/val"  # 源文件夹路径
#     destination_folder = f"G:/seidata/32ft-input/{m}/train2-open"  # 目标文件夹路径
#     num_files_to_move = 8000  # 需要移动的文件数量

    # move_random_files(source_folder, destination_folder, num_files_to_move)
source_folder = f"G:/seidata/32ft-exp2/openset"  # 源文件夹路径
destination_folder = f"G:/seidata/32ft-exp2/test2-mixed"  # 目标文件夹路径
move_random_files(source_folder, destination_folder, 4000)
