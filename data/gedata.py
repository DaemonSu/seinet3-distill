import numpy as np
import h5py
from collections import defaultdict
import os


# --------------------------
# 核心工具函数
# --------------------------
def extract_noise(signal):
    """复现MATLAB的 extractNoise.m 逻辑：3σ异常值过滤 + 高低均值迭代分离"""
    mu = np.mean(signal)
    sigma = np.std(signal)
    mask = (signal >= mu - 3 * sigma) & (signal <= mu + 3 * sigma)
    clean_signal = signal * mask

    # 迭代3次估计高低均值
    high_idx_init = clean_signal >= mu
    low_idx_init = ~high_idx_init
    high_mean = np.mean(clean_signal[high_idx_init]) if np.any(high_idx_init) else mu
    low_mean = np.mean(clean_signal[low_idx_init]) if np.any(low_idx_init) else mu

    for _ in range(3):
        high_idx = np.abs(clean_signal - high_mean) <= np.abs(clean_signal - low_mean)
        low_idx = ~high_idx
        high_mean = np.mean(clean_signal[high_idx]) if np.any(high_idx) else high_mean
        low_mean = np.mean(clean_signal[low_idx]) if np.any(low_idx) else low_mean

    threshold = (high_mean + low_mean) / 2
    medoids = np.where(clean_signal >= threshold, high_mean, low_mean)
    noise_vector = medoids - clean_signal

    return noise_vector, medoids


def find_spatial_dims(length):
    """寻找最接近正方形的因子对（确保a×b=length）"""
    factors = []
    for i in range(1, int(np.sqrt(length)) + 1):
        if length % i == 0:
            factors.append((i, length // i))
    return min(factors, key=lambda x: abs(x[0] - x[1]))


def process_device_samples(device_id, device_code, i_sigs, q_sigs, output_dir, spatial_size):
    """处理单个设备的样本并按指定格式保存"""
    # 创建设备专属目录
    device_dir = os.path.join(output_dir, f"device_{device_code}")
    os.makedirs(device_dir, exist_ok=True)

    # 处理每个样本
    for sample_idx, (i_sig, q_sig) in enumerate(zip(i_sigs, q_sigs), 1):
        # 1. 生成复数IQ信号和基带信号（与仓库格式一致）
        comp_sig = i_sig + 1j * q_sig  # 复数IQ信号
        baseband_sig = np.abs(comp_sig)  # 基带信号（幅度）

        # 2. 噪声处理（与仓库逻辑一致）
        noise, medoids = extract_noise(baseband_sig)

        # 3. 计算FFT特征（实际FFT - 合理信号FFT）
        signal_length = len(baseband_sig)
        actual_fft = np.fft.fftshift(np.fft.fft(comp_sig, n=signal_length))
        rationale_fft = np.fft.fftshift(np.fft.fft(medoids, n=signal_length))
        fft_noise = actual_fft - rationale_fft

        # 4. 构建3通道特征张量（噪声 + FFT实部 + FFT虚部）
        # 格式：(空间维度[0], 空间维度[1], 3)，与仓库输入一致
        feature_tensor = np.zeros((spatial_size[0], spatial_size[1], 3), dtype=np.float32)
        # 通道1：噪声
        feature_tensor[:, :, 0] = noise.reshape(spatial_size, order='F')
        # 通道2：FFT实部
        feature_tensor[:, :, 1] = np.real(fft_noise).reshape(spatial_size, order='F')
        # 通道3：FFT虚部
        feature_tensor[:, :, 2] = np.imag(fft_noise).reshape(spatial_size, order='F')

        # 5. 按指定格式保存样本（device_60_0001.npy）
        sample_name = f"device_{device_code}_{sample_idx:04d}.npy"
        save_path = os.path.join(device_dir, sample_name)
        np.save(save_path, feature_tensor)

        # 打印进度（每100个样本）
        if sample_idx % 100 == 0:
            print(f"设备 {device_code} 已处理 {sample_idx} 个样本")

    print(f"设备 {device_code} 处理完成，共 {len(i_sigs)} 个样本，保存至 {device_dir}")


# --------------------------
# 主流程
# --------------------------
def process_raw_mat_to_samples(mat_file_path, output_root, top_n=100):
    """从原始MAT文件生成按设备分类的样本文件"""
    os.makedirs(output_root, exist_ok=True)

    # 1. 加载原始MAT文件
    print(f"加载原始MAT文件：{mat_file_path}")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            icaoLst = f['icaoLst'][()].flatten()
            rawIMatrix = f['rawIMatrix'][()]
            rawQMatrix = f['rawQMatrix'][()]
    except OSError:
        from scipy.io import loadmat
        mat_data = loadmat(mat_file_path)
        icaoLst = mat_data['icaoLst'].flatten()
        rawIMatrix = mat_data['rawIMatrix']
        rawQMatrix = mat_data['rawQMatrix']

    num_samples = len(icaoLst)
    print(f"原始数据：总样本数={num_samples}，设备总数={len(np.unique(icaoLst))}")

    # 2. 重塑I/Q信号
    rawIMatrix = rawIMatrix.reshape(-1, num_samples, order='F').T  # (样本数, 信号长度)
    rawQMatrix = rawQMatrix.reshape(-1, num_samples, order='F').T
    signal_length = rawIMatrix.shape[1]
    spatial_size = find_spatial_dims(signal_length)
    print(f"信号长度={signal_length}，空间维度={spatial_size[0]}×{spatial_size[1]}")

    # 3. 按设备分组并按消息数排序
    device_groups = defaultdict(lambda: ([], []))  # (I信号列表, Q信号列表)
    for idx in range(num_samples):
        icao = int(icaoLst[idx])
        device_groups[icao][0].append(rawIMatrix[idx, :])
        device_groups[icao][1].append(rawQMatrix[idx, :])

    # 按样本数降序排序，取前100个设备
    sorted_devices = sorted(
        device_groups.items(),
        key=lambda x: len(x[1][0]),
        reverse=True
    )[:top_n]
    print(f"已筛选前{top_n}个消息最多的设备")

    # 4. 处理每个设备并生成样本
    for device_idx, (device_id, (i_sigs, q_sigs)) in enumerate(sorted_devices):
        # 设备编码：00-99
        device_code = f"{device_idx:02d}"
        print(f"\n开始处理设备 {device_code}（原始ID：{device_id}，样本数：{len(i_sigs)}）")

        # 处理并保存该设备的所有样本
        process_device_samples(
            device_id=device_id,
            device_code=device_code,
            i_sigs=i_sigs,
            q_sigs=q_sigs,
            output_dir=output_root,
            spatial_size=spatial_size
        )

    print("\n所有设备处理完成！")


# --------------------------
# 执行入口
# --------------------------
if __name__ == "__main__":
    # 配置参数
    MAT_FILE_PATH = "G:/seidata/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning/adsb_bladerf2_10M_qt0.mat"
    OUTPUT_ROOT = "G:/seidataforCIL/device_samples"  # 根目录，每个设备一个子目录
    TOP_N = 100  # 取前100个消息最多的设备

    # 执行处理
    process_raw_mat_to_samples(
        mat_file_path=MAT_FILE_PATH,
        output_root=OUTPUT_ROOT,
        top_n=TOP_N
    )
