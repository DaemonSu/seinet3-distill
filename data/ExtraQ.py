import numpy as np
import matplotlib.pyplot as plt

# 将数据中的IQ 分别抽取Q，abs(Q) 和 FFT(Q) 并进行可视化

# 读取 SIGMF 文件中的数据
def read_sigmf_file(file_path, num_samples=8000):
    # 使用 sigmf 库读取文件
    with open(file_path, 'rb') as f:
        # 读取 I/Q 数据
        iq_data = np.fromfile(f, dtype=np.complex128)[:num_samples]
    return iq_data

# 绘制 Q 分量、abs(Q) 和 FFT(Q)
def plot_data(iq_data):
    # 提取 Q 分量
    q_component = np.imag(iq_data)

    # 计算 abs(Q)
    abs_q = np.abs(q_component)

    # 计算 FFT(Q)
    fft_q = np.fft.fft(q_component)

    # 设置绘图布局
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # 绘制 Q 分量
    axs[0].plot(q_component, label='Q component', color='blue')
    axs[0].set_title('Q Component')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # 绘制 abs(Q)
    axs[1].plot(abs_q, label='abs(Q)', color='green')
    axs[1].set_title('abs(Q) Magnitude')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()

    # 绘制 FFT(Q)
    axs[2].plot(np.abs(fft_q), label='FFT of Q', color='red')
    axs[2].set_title('FFT(Q) Magnitude')
    axs[2].set_xlabel('Frequency Bin')
    axs[2].set_ylabel('Magnitude')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# 示例：读取文件并绘制图形
if __name__ == "__main__":
    file_path = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/62ft/WiFi_air_X310_3123D7B_62ft_run1.sigmf-data"
    iq_data = read_sigmf_file(file_path)
    plot_data(iq_data)
