import numpy as np
import os

def load_and_check_shape(file_path):
    """
    加载 IQ 数据文件并打印其形状
    :param file_path: IQ 数据文件路径 (.npy)
    :return: IQ 数据的 NumPy 数组
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None

    try:
        data = np.load(file_path)
        print(f"✅ 文件加载成功: {file_path}")
        print(f"📏 数据形状: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

# 示例使用
file_path = "F:/seidata/IQdata/train/device_05_0551.npy"  # 替换为你的文件路径
iq_data = load_and_check_shape(file_path)
