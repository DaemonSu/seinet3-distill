import numpy as np
from scipy.io import loadmat
from collections import defaultdict
import os


def extract_noise(signal):
    """
    实现与MATLAB extractNoise 类似的噪声提取逻辑
    :param signal: 输入信号（1D数组）
    :return: noise: 提取的噪声
             rational_signal: 分离出的合理信号
    """
    # 3σ异常值过滤
    mu = np.mean(signal)
    sigma = np.std(signal)
    # 生成掩码：保留均值±3σ范围内的数据
    mask = (signal >= mu - 3 * sigma) & (signal <= mu + 3 * sigma)
    clean_signal = signal * mask  # 过滤异常值后的信号

    # 迭代估计高低均值，分离合理信号与噪声
    # 初始高低均值设置
    high_mean = np.mean(clean_signal[clean_signal >= mu]) if np.any(clean_signal >= mu) else mu
    low_mean = np.mean(clean_signal[clean_signal < mu]) if np.any(clean_signal < mu) else mu

    # 迭代优化高低均值（3次迭代，与MATLAB逻辑一致）
    for _ in range(3):
        high_indices = np.abs(clean_signal - high_mean) <= np.abs(clean_signal - low_mean)
        low_indices = ~high_indices
        high_mean = np.mean(clean_signal[high_indices]) if np.any(high_indices) else high_mean
        low_mean = np.mean(clean_signal[low_indices]) if np.any(low_indices) else low_mean

    # 确定合理信号阈值（高低均值的中点）
    threshold = (high_mean + low_mean) / 2
    # 生成合理信号（类似MATLAB中的 rationalSignal）
    rational_signal = np.where(clean_signal >= threshold, high_mean, low_mean)
    # 噪声 = 合理信号 - 原始干净信号（与MATLAB逻辑一致）
    noise = rational_signal - clean_signal

    return noise, rational_signal


def process_raw_data(mat_file_path, output_dir, top_n=100, process_noise=True):
    """
    处理原始MAT文件，按设备ID分组并保存前N个设备的IQ信号，增加噪声处理
    保存格式：二维数组 (2, 总点数)，其中第一行是I信号，第二行是Q信号
             若开启噪声处理，同时保存去噪后的信号和噪声
    """
    os.makedirs(output_dir, exist_ok=True)

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
        device_groups_i[icao.item()].append(rawIMatrix[idx])
        device_groups_q[icao.item()].append(rawQMatrix[idx])

    # 按样本数排序并选择前N个设备
    print(f"按样本数排序，选取前{top_n}个设备...")
    sorted_devices = sorted(
        device_groups_i.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_n]

    # 保存每个设备的IQ信号及噪声（若开启处理）
    for idx, (device_id, i_signals) in enumerate(sorted_devices):
        q_signals = device_groups_q[device_id]

        # 验证数据一致性
        assert len(i_signals) == len(q_signals), f"设备{device_id}的I/Q样本数不匹配"
        signal_length = len(i_signals[0]) if i_signals else 0
        assert all(len(sig) == signal_length for sig in i_signals), f"设备{device_id}的I信号长度不一致"
        assert all(len(sig) == signal_length for sig in q_signals), f"设备{device_id}的Q信号长度不一致"

        # 对每个样本进行噪声处理
        processed_i = []
        processed_q = []
        noise_i_list = []
        noise_q_list = []

        for i_sig, q_sig in zip(i_signals, q_signals):
            if process_noise:
                # 处理I信号噪声
                noise_i, rational_i = extract_noise(i_sig)
                # 处理Q信号噪声
                noise_q, rational_q = extract_noise(q_sig)

                processed_i.append(rational_i)  # 去噪后的I信号
                processed_q.append(rational_q)  # 去噪后的Q信号
                noise_i_list.append(noise_i)  # I信号噪声
                noise_q_list.append(noise_q)  # Q信号噪声
            else:
                # 不处理噪声，直接使用原始信号
                processed_i.append(i_sig)
                processed_q.append(q_sig)

        # 拼接所有样本（原始/去噪信号）
        all_i = np.concatenate(processed_i)
        all_q = np.concatenate(processed_q)
        iq_2d = np.row_stack((all_i, all_q))

        # 保存去噪后的IQ信号
        save_path = os.path.join(output_dir, f"device_{idx:02d}_iq_denoised.npy")
        np.save(save_path, iq_2d)

        # 若开启噪声处理，保存噪声信号
        if process_noise:
            all_noise_i = np.concatenate(noise_i_list)
            all_noise_q = np.concatenate(noise_q_list)
            noise_2d = np.row_stack((all_noise_i, all_noise_q))
            noise_save_path = os.path.join(output_dir, f"device_{idx:02d}_noise.npy")
            np.save(noise_save_path, noise_2d)

        # 打印信息
        num_samples = len(i_signals)
        print(f"设备 {idx:02d} (ID: {device_id}) 处理完成，样本数: {num_samples}, "
              f"总点数: {iq_2d.shape[1]}, 去噪信号保存至: {save_path}")
        if process_noise:
            print(f"  噪声信号保存至: {noise_save_path}")


if __name__ == "__main__":
    # 配置参数
    MAT_FILE_PATH = "G:/seidata/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning/adsb_bladerf2_10M_qt0.mat"
    OUTPUT_DIR = "G:/seidataforCIL/IQArray3_with_noise"
    TOP_N = 100  # 选取前100个设备
    PROCESS_NOISE = True  # 是否开启噪声处理

    # 执行处理
    process_raw_data(MAT_FILE_PATH, OUTPUT_DIR, TOP_N, PROCESS_NOISE)
    print("处理完成！")
