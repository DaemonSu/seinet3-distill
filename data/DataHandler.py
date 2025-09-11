import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------
#  1️⃣ 读取 IQ 数据文件
# ---------------------------
DATA_PATH = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ft-exp"  # 文件夹路径，存放 16 个设备的 IQ 数据文件
SAMPLE_SIZE = 7000  # 每条数据包含 7000 个 IQ 采样点
STRIDE = 7000       # 步长（可调）
TRAIN_DEVICES = 8
VAL_DEVICES = 4
OPENSET_DEVICES = 4

SAVE_PATH = "G:/seidata/32ft-exp2"  # 训练数据保存目录


def load_iq_data(file_path):

    iq_data =  np.fromfile(file_path, dtype=np.complex128)
    if np.iscomplexobj(iq_data):  # 检测是否是复数格式


        image= torch.from_numpy(iq_data.real)
        # 拆分 I/Q 组成 (N,2)
        q_abs = torch.abs(image)
        q_fft = torch.fft.fft(image).abs()
        # iq_data = np.vstack((iq_data.imag,q_abs,q_fft))
        iq_data = np.column_stack((iq_data.real, q_abs, q_fft))

    return iq_data


def load_alliq_data(file_path):
    # 从文件中读取复数数据
    iq_data = np.fromfile(file_path, dtype=np.complex128)

    if np.iscomplexobj(iq_data):
        # 使用完整复数 IQ 形式
        iq_tensor = torch.from_numpy(iq_data)  # shape: (N, )

        i_part = iq_tensor.real
        q_part = iq_tensor.imag
        abs_iq = torch.abs(iq_tensor)
        fft_abs = torch.fft.fft(iq_tensor).abs()

        # 合并为 (N, 4) 特征：I, Q, |IQ|, |FFT(IQ)|
        features = torch.stack([i_part, q_part, abs_iq, fft_abs], dim=1)
        return features
# ---------------------------
#  2️⃣ 处理数据 & 划分数据集
# ---------------------------
class SEIDataset(Dataset):
    def __init__(self, iq_data, labels):
        self.iq_data = torch.tensor(iq_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.iq_data)

    def __getitem__(self, idx):
        return self.iq_data[idx], self.labels[idx]


def process_data():
    iq_samples = []  # 存放 IQ 数据
    labels = []  # 存放设备类别
    device_files = sorted(os.listdir(DATA_PATH))  # 读取 16 个设备文件

    # 确保设备数据排序
    assert len(device_files) == 16, "数据集应包含 16个设备文件"

    # 设备编号分配
    train_files = device_files[:TRAIN_DEVICES]
    val_files = device_files[TRAIN_DEVICES:TRAIN_DEVICES + VAL_DEVICES]
    openset_files = device_files[TRAIN_DEVICES + VAL_DEVICES:]

    def process_device(files, save_folder, label_offset=0):

        save_path = os.path.join(SAVE_PATH, save_folder)
        os.makedirs(save_path, exist_ok=True)

        for device_idx, file in enumerate(files):
            iq_data = load_iq_data(os.path.join(DATA_PATH, file))
            total_points = iq_data.shape[0]  # IQ 点数
            device_id = device_idx + label_offset

            # # 计算可分割的样本数 N
            num_samples = (total_points - SAMPLE_SIZE) // STRIDE + 1

            for i in range(num_samples):
                start = i * STRIDE
                segment = iq_data[start: start + SAMPLE_SIZE]

                # 生成文件名：device_01_0001.npy
                filename = f"device_{device_id:02d}_{i:04d}.npy"
                np.save(os.path.join(save_path, filename), segment)

            print(f"✅ 设备 {device_id} 处理完成: {num_samples} 片段")

    # 划分训练、验证、开集测试数据
    process_device(train_files,"train", label_offset=0)
    process_device(val_files,"val", label_offset=TRAIN_DEVICES)
    process_device(openset_files,"openset", label_offset=TRAIN_DEVICES + VAL_DEVICES)



# ---------------------------
#  4️⃣ 运行数据预处理
# ---------------------------
if __name__ == "__main__":
    process_data()

