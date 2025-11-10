import numpy as np
from scipy.io import loadmat
from collections import defaultdict
import os

import h5py
def process_raw_data(mat_file_path, output_dir, top_n=100):
    """
    处理原始MAT文件，按设备ID分组并保存前100个设备的IQ信号
    保存格式：二维数组 (2, 总点数)，其中第一行是I信号，第二行是Q信号
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # with h5py.File(mat_file_path, 'r') as f:
    #     all_keys = list(f.keys())
    #     print("文件中的所有变量键：", all_keys)
    #     print("\n---- 遍历所有键和对应值 ----\n")
    #
    #     for key in all_keys:
    #         # 获取当前键对应的数据对象（可能是数据集或群组）
    #         data_obj = f[key]
    #
    #         # 判断是否为数据集（实际存储数据的对象）
    #         if isinstance(data_obj, h5py.Dataset):
    #             # 转换为 numpy 数组（[()] 用于读取整个数据集）
    #             data = data_obj[()]
    #
    #             # 处理字符串（MATLAB 字符串在 h5py 中可能以字节串存储）
    #             if data.dtype.kind in ['S', 'U']:  # 字符串类型
    #                 data = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in data.flat]).reshape(
    #                     data.shape)
    #
    #             # 打印键和值的信息
    #             print(f"键: {key}")
    #             print(f"  数据类型: {data.dtype}")
    #             print(f"  形状: {data.shape}")
    #             print(f"  部分数据示例: {data.flat[:5]}...\n")  # 打印前5个元素
    #
    #         # 如果是群组（嵌套结构），可以递归遍历（可选）
    #         elif isinstance(data_obj, h5py.Group):
    #             print(f"键: {key} (嵌套群组，包含子键: {list(data_obj.keys())})\n")

    # 加载原始数据
    print(f"加载原始数据: {mat_file_path}")
    mat_data = loadmat(mat_file_path)

    # 提取关键数据（与MATLAB代码对应）
    icaoLst = mat_data['icaoLst'].flatten()  # 设备ID列表
    rawIMatrix = mat_data['rawIMatrix']  # I信号矩阵
    rawQMatrix = mat_data['rawQMatrix']  # Q信号矩阵
    msgIdLst = mat_data['msgIdLst']  # 用于计算样本数量
    num_samples = len(msgIdLst)  # 样本总数

    # 重塑矩阵为 (样本数 x 信号长度) 格式（与MATLAB reshape和转置一致）
    rawIMatrix = rawIMatrix.reshape(-1, num_samples).T  # 形状: (样本数, 信号长度)
    rawQMatrix = rawQMatrix.reshape(-1, num_samples).T  # 形状: (样本数, 信号长度)

    # 按设备ID分组
    device_groups_i = defaultdict(list)  # 存储每个设备的I信号
    device_groups_q = defaultdict(list)  # 存储每个设备的Q信号
    for idx, icao in enumerate(icaoLst):
        device_groups_i[icao.item()].append(rawIMatrix[idx])  # 每个样本的I信号（长度1218）
        device_groups_q[icao.item()].append(rawQMatrix[idx])  # 每个样本的Q信号（长度1218）

    # 按样本数排序并选择前100个设备
    print(f"按样本数排序，选取前{top_n}个设备...")
    sorted_devices = sorted(
        device_groups_i.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_n]

    # 保存每个设备的IQ信号为二维数组（I和Q分行）
    for idx, (device_id, i_signals) in enumerate(sorted_devices):
        q_signals = device_groups_q[device_id]  # 获取对应设备的Q信号

        # 验证I和Q信号数量及长度匹配
        assert len(i_signals) == len(q_signals), f"设备{device_id}的I/Q样本数不匹配"
        signal_length = len(i_signals[0]) if i_signals else 0
        assert all(len(sig) == signal_length for sig in i_signals), f"设备{device_id}的I信号长度不一致"
        assert all(len(sig) == signal_length for sig in q_signals), f"设备{device_id}的Q信号长度不一致"

        # 拼接所有样本的I和Q信号：(2, 总点数)，其中总点数 = 样本数 × 单样本长度
        # 例如6119个样本×1218长度 → 形状为(2, 6119×1218) = (2, 7452942)
        all_i = np.concatenate(i_signals)  # 形状: (总点数,)
        all_q = np.concatenate(q_signals)  # 形状: (总点数,)
        iq_2d = np.row_stack((all_i, all_q))  # 合并为二维数组，第一行I，第二行Q

        # 保存为npy文件
        save_path = os.path.join(output_dir, f"device_{idx:02d}_iq.npy")
        np.save(save_path, iq_2d)

        # 打印信息
        num_samples = len(i_signals)
        print(f"保存设备 {idx:02d} (ID: {device_id}) 到 {save_path}，样本数: {num_samples}, "
              f"单样本长度: {signal_length}, 总点数: {iq_2d.shape[1]}, 数组形状: {iq_2d.shape}")


if __name__ == "__main__":
    # 配置参数
    # MAT_FILE_PATH = "G:/seidata/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning/adsb-107loaded.mat"  # 输入MAT文件路径
    MAT_FILE_PATH = "G:/seidata/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning/adsb_bladerf2_10M_qt0.mat"  # 输入MAT文件路径
    OUTPUT_DIR = "G:/seidataforCIL/IQArray3"  # 输出目录
    TOP_N = 100  # 选取前100个设备

    # 执行处理
    process_raw_data(MAT_FILE_PATH, OUTPUT_DIR, TOP_N)
    print("处理完成！")
