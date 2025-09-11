import numpy as np
import matplotlib
matplotlib.use('pdf')  # 使用 PDF 后端（矢量图支持）
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 设置字体为论文风格（如 IEEE、Springer）
plt.rcParams.update({
    "pdf.fonttype": 42,           # 保证文字可选中（矢量字体）
    "ps.fonttype": 42,
    "font.family": "Times New Roman",
    "font.size": 12
})

# 读取信号数据
data = np.load("G:/seidata/26ft-exp/train/device_01_0001.npy")  # shape: (7000, 3)
imag = data[:, 0]
q_abs = data[:, 1]
q_fft = data[:, 2]
x = np.arange(data.shape[0])

# 禁用科学计数法
formatter = ScalarFormatter(useOffset=False, useMathText=False)
formatter.set_scientific(False)

# 创建图形
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(x, imag, color='blue')
plt.title("Imaginary Part")
plt.ylabel("Amplitude")
plt.gca().xaxis.set_major_formatter(formatter)

plt.subplot(3, 1, 2)
plt.plot(x, q_abs, color='green')
plt.title("Abs(Imag)")
plt.ylabel("Amplitude")
plt.gca().xaxis.set_major_formatter(formatter)

plt.subplot(3, 1, 3)
plt.plot(x, q_fft, color='red')
plt.title("FFT Abs(Imag)")
plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.gca().xaxis.set_major_formatter(formatter)

# 自动调整子图间距
plt.tight_layout()

# 保存为矢量图PDF
plt.savefig("signal_visualization.pdf", format='pdf', dpi=300)

# 若仅调试阶段显示，可开启此行
# plt.show()
