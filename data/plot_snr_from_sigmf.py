import numpy as np
import os
import matplotlib.pyplot as plt


def read_iq_data(file_path):
    """
    读取 .sigmf-data 文件中的复数 IQ 数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    iq_data = np.fromfile(file_path, dtype=np.complex64)
    return iq_data


def estimate_snr(iq_data, sample_rate=5e6, signal_ratio=0.5, noise_window=0.02):
    """
    估算 SNR：
    - 中心区域为信号段
    - 两端作为噪声段
    返回估计的 SNR（dB）以及信号段起止位置
    """
    iq_data = np.fromfile(file_path, dtype=np.complex64)
    print("IQ mean:", np.mean(iq_data))
    print("IQ std:", np.std(iq_data))
    print("Max abs:", np.max(np.abs(iq_data)))



    total_len = len(iq_data)
    signal_len = int(total_len * signal_ratio)
    noise_len = int(noise_window * sample_rate)

    signal_start = total_len // 2 - signal_len // 2
    signal_end = signal_start + signal_len

    signal = iq_data[signal_start:signal_end]
    noise = np.concatenate([iq_data[:noise_len], iq_data[-noise_len:]])

    # 去除异常值
    signal = signal[np.isfinite(signal)]
    noise = noise[np.isfinite(noise)]

    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)

    if noise_power == 0 or signal_power == 0 or np.isnan(noise_power) or np.isnan(signal_power):
        snr_db = float('nan')
    else:
        snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db, signal_start, signal_end


def plot_spectrum(iq_data, sample_rate, signal_start, signal_end, snr_db, title="Frequency Spectrum"):
    """
    绘制频谱图，并高亮信号段
    """
    fft_data = np.fft.fftshift(np.fft.fft(iq_data))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), d=1/sample_rate))
    power_db = 10 * np.log10(np.abs(fft_data)**2 + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs / 1e6, power_db, color='steelblue', label='Spectrum')

    # 估计信号带宽
    bandwidth_hz = (signal_end - signal_start) / len(iq_data) * sample_rate
    plt.axvspan(-bandwidth_hz / 2 / 1e6, bandwidth_hz / 2 / 1e6,
                color='orange', alpha=0.3, label='Estimated Signal Band')

    plt.title(f"{title}\nEstimated SNR: {snr_db:.2f} dB")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/62ft/WiFi_air_X310_3123D7B_62ft_run1.sigmf-data"

    sample_rate = 5e6  # 固定采样率
    iq_data = read_iq_data(file_path)

    snr_db, signal_start, signal_end = estimate_snr(iq_data, sample_rate)
    print(f"Estimated SNR: {snr_db:.2f} dB")

    plot_spectrum(iq_data, sample_rate, signal_start, signal_end, snr_db)
