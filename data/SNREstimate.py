import numpy as np
import os


def estimate_snr_from_sigmf_data(file_path, sample_rate=5e6, signal_ratio=0.5, window_duration=0.02):
    """
    估算给定 SigMF .sigmf-data 文件的 SNR（单位：dB）
    - signal_ratio: 信号占比，用于分离信号段与噪声段
    - window_duration: 用于计算的窗口大小（秒）

    返回:
    - snr_db: 估算的信噪比（dB）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 读取 IQ 数据
    iq_data = np.fromfile(file_path, dtype=np.complex64)

    # 去除NaN/Inf和极端值
    iq_data = iq_data[~np.isnan(iq_data)]
    iq_data = iq_data[~np.isinf(iq_data)]
    iq_data = iq_data[np.abs(iq_data) < 1e3]  # 阈值可调整

    total_samples = len(iq_data)
    window_size = int(sample_rate * window_duration)

    if total_samples < 2 * window_size:
        raise ValueError("Data too short for SNR estimation.")

    # 中间为信号段，前后为噪声段（假设中心为主信号）
    signal_start = int(total_samples * (0.5 - signal_ratio / 2))
    signal_end = int(total_samples * (0.5 + signal_ratio / 2))

    signal_segment = iq_data[signal_start:signal_end]
    noise_segment = np.concatenate([
        iq_data[:window_size],
        iq_data[-window_size:]
    ])

    # 再次清洗
    def clean(x):
        x = x[~np.isnan(x)]
        x = x[~np.isinf(x)]
        return x[np.abs(x) < 1e3]

    signal_segment = clean(signal_segment)
    noise_segment = clean(noise_segment)

    # 计算功率
    signal_power = np.mean(np.abs(signal_segment) ** 2)
    noise_power = np.mean(np.abs(noise_segment) ** 2)

    if noise_power == 0 or np.isnan(signal_power) or np.isnan(noise_power):
        print("Invalid power values detected.")
        return float('nan')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


# 使用示例
if __name__ == "__main__":
    path = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/62ft/WiFi_air_X310_3123D7B_62ft_run1.sigmf-data"
    # path = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/14ft/WiFi_air_X310_3123D7B_14ft_run1.sigmf-data"
    snr = estimate_snr_from_sigmf_data(path)
    print(f"Estimated SNR: {snr:.2f} dB")
