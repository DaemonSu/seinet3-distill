import os
import shutil
import random
import numpy as np


def move_random_samples(source_root, target_dir, device_codes, num_per_device):
    """
    随机移动指定设备的样本到目标文件夹

    参数:
        source_root: 源目录根路径（包含device_XX子目录）
        target_dir: 目标文件夹路径
        device_codes: 要处理的设备编码列表（如['00', '01', '05']）
        num_per_device: 每个设备需要移动的样本数量
    """
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标文件夹: {target_dir}（已确保存在）")

    # 遍历每个指定设备
    for code in device_codes:
        # 设备源目录（如source_root/device_00）
        device_source = os.path.join(source_root, f"device_{code}")

        # 检查设备目录是否存在
        if not os.path.exists(device_source):
            print(f"警告：设备 {code} 的源目录 {device_source} 不存在，跳过该设备")
            continue

        # 获取该设备下所有样本文件（筛选.npy文件且符合命名格式）
        all_samples = []
        for filename in os.listdir(device_source):
            # 匹配格式：device_XX_YYYY.npy
            if (filename.endswith(".npy") and
                    filename.startswith(f"device_{code}_") and
                    len(filename) == len(f"device_{code}_0000.npy")):
                all_samples.append(filename)

        # 检查样本数量
        total_available = len(all_samples)
        if total_available == 0:
            print(f"警告：设备 {code} 没有可用样本，跳过")
            continue

        # 确定实际移动数量（如果可用样本不足，取全部）
        actual_num = min(num_per_device, total_available)
        if actual_num < num_per_device:
            print(f"警告：设备 {code} 仅找到 {total_available} 个样本，将移动全部 {actual_num} 个")

        # 随机选择样本（固定随机种子，确保结果可复现；如需每次不同，可删除seed参数）
        random.seed(42)  # 可选：移除该行则每次随机结果不同
        selected_samples = random.sample(all_samples, actual_num)

        # 移动选中的样本
        moved_count = 0
        for sample in selected_samples:
            src_path = os.path.join(device_source, sample)
            dst_path = os.path.join(target_dir, sample)

            # 避免目标文件已存在（可选：覆盖/跳过）
            if os.path.exists(dst_path):
                print(f"跳过已存在文件：{sample}")
                continue

            # 移动文件
            shutil.move(src_path, dst_path)
            moved_count += 1

        print(f"设备 {code} 处理完成：成功移动 {moved_count}/{actual_num} 个样本")

    print("\n所有设备处理完毕！")


if __name__ == "__main__":
    # --------------------------
    # 配置参数（根据需求修改）
    # --------------------------
    SOURCE_ROOT = "G:/seidataforCIL/device_samples"  # 源目录根路径（包含device_XX子目录）
    TARGET_DIR = "G:/seidataforCIL/train-openset"  # 目标文件夹路径
    # DEVICE_CODES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']  # 要处理的设备编码（两位数字符串）

    # DEVICE_CODES = ['20', '21', '22', '23', '24', '25', '26', '27', '22', '29','30', '31', '32', '33', '34', '35', '36', '37', '33', '39']  # 要处理的设备编码（两位数字符串）

    DEVICE_CODES = ['70', '71', '72', '73', '74', '75', '76', '77', '78', '79']  # 要处理的设备编码（两位数字符串）

    # DEVICE_CODES = ['98']
    # DEVICE_CODES = ['90', '91', '92', '93', '94', '95', '96', '97', '98', '99']  # 要处理的设备编码（两位数字符串）
    NUM_PER_DEVICE = 520  # 每个设备需要移动的样本数量

    # 执行移动
    move_random_samples(
        source_root=SOURCE_ROOT,
        target_dir=TARGET_DIR,
        device_codes=DEVICE_CODES,
        num_per_device=NUM_PER_DEVICE
    )
