import os
import re


def count_device_files(folder_path):
    """
    统计文件夹中各设备的文件数量
    :param folder_path: 文件夹路径
    :return: 设备编号到文件数量的字典
    """
    # 正则表达式模式：匹配device_xx_xxxx格式的文件名
    pattern = r'^device_(\d+)_\d+\.\w+$'

    # 用于存储每个设备的文件数量
    device_counts = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否是文件（不是文件夹）
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # 使用正则表达式匹配文件名
            match = re.match(pattern, filename)
            if match:
                # 提取设备编号
                device_id = match.group(1)
                # 更新设备文件计数
                device_counts[device_id] = device_counts.get(device_id, 0) + 1

    return device_counts


def main():
    # 让用户输入文件夹路径
    folder_path =f"G:/seidataforCIL/train-add1"

    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 路径 '{folder_path}' 不存在")
        return

    # 检查是否是文件夹
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return

    # 统计设备文件数量
    device_counts = count_device_files(folder_path)

    # 输出统计结果
    print("\n设备文件数量统计结果:")
    print("======================")
    if device_counts:
        for device_id, count in sorted(device_counts.items()):
            print(f"设备 {device_id}: {count} 个文件")
    else:
        print("未找到符合格式的文件")


if __name__ == "__main__":
    main()
