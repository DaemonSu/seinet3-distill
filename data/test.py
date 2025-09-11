import numpy as np
import matplotlib.pyplot as plt


import numpy as np

def estimate_snr_oracle(iq_data, noise_power_baseline):
    total_power = np.mean(np.abs(iq_data)**2)

    # 估计信号功率 = 总功率 - 噪声功率基线
    signal_power = max(total_power - noise_power_baseline, 1e-20)
    noise_power = noise_power_baseline

    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db, signal_power, noise_power


if __name__ == "__main__":
    # 测试示例
    # 这里假设你测得设备噪声功率约为1e-7
    noise_baseline = 1e-7
    file_path = r"G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ft-6db/WiFi_air_X310_3123D52_32ft_run1.sigmf-data"
    # 载入IQ数据
    iq_data = np.fromfile(file_path, dtype=np.complex128)

    snr, sig_p, noise_p = estimate_snr_oracle(iq_data, noise_baseline)
    print(f"Estimated SNR: {snr:.2f} dB")
    print(f"Signal power: {sig_p:.6e}")
    print(f"Noise power baseline (prior): {noise_p:.6e}")


