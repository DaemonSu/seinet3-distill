import os
import numpy as np

DATA_PATH = "G:/博士科研/一汽课题/论文阅读/综述/131/neu_m044q5210/KRI-16Devices-RawData/32ft-exp"
SAMPLE_SIZE = 7000
STRIDE = 7000
TRAIN_DEVICES = 8
VAL_DEVICES = 4
OPENSET_DEVICES = 4
SAVE_PATH = "G:/seidata/32ft-input"

def load_features(file_path, mode="Q_ABS_FFT", out_dtype=None):
    iq = np.fromfile(file_path, dtype=np.complex128)
    if not np.iscomplexobj(iq):
        raise ValueError(f"{file_path} 不是复数 IQ 数据")

    i = iq.real
    q = iq.imag
    abs_iq = np.abs(iq)
    fft_i = np.abs(np.fft.fft(i))
    fft_q = np.abs(np.fft.fft(q))
    fft_iq = np.abs(np.fft.fft(iq))

    if mode == "Q":
        features = q[:, None]                       # (N,1)
    elif mode == "Q_ABS":
        features = np.stack([q, np.abs(q)], axis=1) # (N,2)
    elif mode == "Q_ABS_FFT":
        features = np.stack([q, np.abs(q), fft_q], axis=1) # (N,3)
    elif mode == "I_ABS_FFT":
        features = np.stack([i, np.abs(i), fft_i], axis=1) # (N,3)
    elif mode == "IQ_ABS_FFT":
        features = np.stack([i, q, abs_iq, fft_iq], axis=1) # (N,4)
    else:
        raise ValueError(f"未知 mode: {mode}")

    if out_dtype is not None:
        features = features.astype(out_dtype)

    return features

def process_data(mode="Q_ABS_FFT", out_dtype=None):
    device_files = sorted(os.listdir(DATA_PATH))
    assert len(device_files) == 16, "数据集应包含16个设备文件"

    train_files = device_files[:TRAIN_DEVICES]
    val_files = device_files[TRAIN_DEVICES:TRAIN_DEVICES + VAL_DEVICES]
    openset_files = device_files[TRAIN_DEVICES + VAL_DEVICES:]

    def process_device(files, save_folder, label_offset=0):
        save_path = os.path.join(SAVE_PATH, mode, save_folder)
        os.makedirs(save_path, exist_ok=True)
        for device_idx, file in enumerate(files):
            features = load_features(os.path.join(DATA_PATH, file), mode=mode, out_dtype=out_dtype)
            total_points = features.shape[0]
            device_id = device_idx + label_offset
            num_samples = (total_points - SAMPLE_SIZE) // STRIDE + 1
            for i in range(num_samples):
                start = i * STRIDE
                segment = features[start: start + SAMPLE_SIZE]  # shape: (SAMPLE_SIZE, C)
                filename = f"device_{device_id:02d}_{i:04d}.npy"
                np.save(os.path.join(save_path, filename), segment)
            print(f"[{mode} | dtype={features.dtype}] 设备 {device_id} 完成: {num_samples} 样本, 单片尺寸: {segment.nbytes} bytes")

    process_device(train_files, "train", label_offset=0)
    process_device(val_files, "val", label_offset=TRAIN_DEVICES)
    process_device(openset_files, "openset", label_offset=TRAIN_DEVICES + VAL_DEVICES)

# 使用示例：
# 生成与原来一致的（float64）：
# process_data(mode="Q_ABS_FFT", out_dtype=None)

# 或者，生成 float32（文件体积约为一半，推荐用于训练）：
# process_data(mode="Q_ABS_FFT", out_dtype=np.float32)
if __name__ == "__main__":
    modes = ["Q", "Q_ABS", "Q_ABS_FFT", "I_ABS_FFT", "IQ_ABS_FFT"]
    for m in modes:
        process_data(mode=m, out_dtype=None)
