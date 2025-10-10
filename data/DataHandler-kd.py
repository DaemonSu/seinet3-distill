import os
import numpy as np
import shutil

from scipy.signal import stft

# ---------------------------
# 配置参数
# ---------------------------
INPUT_DIR = "G:/seidataforCIL/IQArray3_without_noise"  # 存放device_XX_iq.npy的文件夹
OUTPUT_DIR = "G:/seidataforCIL"  # 处理后数据保存根目录
TRAIN_DEVICES = 20  # 训练设备数量
OPENSET_DEVICES = 10  # 开集设备数量
ADD_GROUP_SIZE = 20  # 每个add组的设备数量
SAMPLE_SIZE = 7000  # 每个样本的采样点数
STRIDE = 7000  # 样本分割步长（无重叠）


# ---------------------------
# 辅助函数：获取设备文件列表并按大小排序
# ---------------------------
def get_sorted_device_files(input_dir):
    """获取所有设备IQ文件并按文件大小降序排序"""
    device_files = []

    # 遍历目录下所有device_XX_iq.npy文件
    for filename in os.listdir(input_dir):
        if filename.startswith("device_") and filename.endswith("_iq_denoised.npy"):
            file_path = os.path.join(input_dir, filename)
            # 获取文件大小（字节）
            file_size = os.path.getsize(file_path)
            # 提取设备ID
            device_id = int(filename.split("_")[1])
            device_files.append((file_path, device_id, file_size))

    # 按文件大小降序排序（先处理大文件）
    device_files.sort(key=lambda x: x[2], reverse=True)
    return device_files


def extract_features(iq_file_path):
    """只提取Q域和绝对值（不做FFT），FFT在分段后再做"""
    try:
        iq_data = np.load(iq_file_path,allow_pickle=True)
        i_part = iq_data[0, :]  # Q信号
        q_part = iq_data[1, :]  # Q信号
        # q_abs = np.abs(q_part)
        # 先返回 (N,2)，第三列FFT留空
        features = np.column_stack((i_part,q_part))
        return features
    except Exception as e:
        print(f"处理文件 {iq_file_path} 出错: {str(e)}")
        return None
# ---------------------------
# 样本生成函数（支持数据对齐）
# ---------------------------
import numpy as np
import pywt

def generate_samples(features, max_samples=None, cwt_wavelet='cmor1.5-1.0', cwt_scales=None):
    """
    从特征矩阵生成样本，并在每段内计算FFT + CWT（时频图）
    输出特征长度和Q一致
    """
    total_points = features.shape[0]
    if total_points < SAMPLE_SIZE:
        return []

    if cwt_scales is None:
        cwt_scales = np.arange(1, 64)  # 默认尺度

    max_possible = total_points // SAMPLE_SIZE  # 不重叠
    if max_samples is not None:
        num_samples = min(max_possible, max_samples)
    else:
        num_samples = max_possible

    samples = []
    for i in range(num_samples):
        start = i * SAMPLE_SIZE
        end = start + SAMPLE_SIZE

        segment = features[start:end, :]
        i_segment = segment[:, 0]
        q_segment = segment[:, 1]

        # 复包络 FFT
        # iq_segment = i_segment + 1j * q_segment
        iq_fft = np.fft.fft(q_segment)
        fft_abs = np.abs(iq_fft)

        # 构造样本特征 (SAMPLE_SIZE, 6)
        segment_features = np.column_stack((
            q_segment,
            np.abs(q_segment),
            fft_abs
        ))
        segment_features = (segment_features - np.min(segment_features, axis=0)) / (
                    np.max(segment_features, axis=0) - np.min(segment_features, axis=0) + 1e-8)

        samples.append(segment_features)

    return samples



# ---------------------------
# 主处理函数
# ---------------------------
def process_all_devices():
    # 1. 获取排序后的设备文件列表
    sorted_files = get_sorted_device_files(INPUT_DIR)
    if not sorted_files:
        print("未找到设备IQ文件")
        return

    total_devices = len(sorted_files)
    print(f"找到 {total_devices} 个设备文件，按大小降序处理")

    # 2. 划分设备组
    # 训练集设备（前20个）
    train_files = sorted_files[:TRAIN_DEVICES]
    # 开集设备（接下来10个）
    # openset_files = sorted_files[TRAIN_DEVICES: TRAIN_DEVICES + OPENSET_DEVICES]
    # 剩余设备（用于add组）
    remaining_files = sorted_files[TRAIN_DEVICES:]

    # 3. 计算train和openset的对齐样本数（取最小可能的最大样本数）
    # 先统计所有训练设备能生成的最大样本数
    train_max_samples = []
    for file_path, _, _ in train_files:
        features = extract_features(file_path)
        if features is None:
            continue
        total_points = features.shape[0]
        max_possible = (total_points - SAMPLE_SIZE) // STRIDE + 1
        train_max_samples.append(max_possible)

    # 取训练集中的最小样本数作为对齐基准
    if not train_max_samples:
        print("没有有效的训练设备数据")
        return
    align_samples = min(train_max_samples)
    print(f"数据对齐：train和openset每组设备生成 {align_samples} 个样本")

    # 4. 处理训练集
    train_dir = os.path.join(OUTPUT_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)
    print(f"\n开始处理训练集（{len(train_files)}个设备）")

    for idx, (file_path, device_id, _) in enumerate(train_files):
        features = extract_features(file_path)
        if features is None:
            continue

        # 生成指定数量的样本（对齐）
        samples = generate_samples(features, max_samples=align_samples)
        if not samples:
            print(f"设备 {device_id} 无法生成足够样本，跳过")
            continue

        # 保存样本
        for i, sample in enumerate(samples):
            save_path = os.path.join(train_dir, f"device_{device_id:02d}_{i:04d}.npy")
            np.save(save_path, sample)

        print(f"设备 {device_id} 处理完成，生成 {len(samples)} 个样本")

    # 5. 处理开集
    openset_dir = os.path.join(OUTPUT_DIR, "openset")
    os.makedirs(openset_dir, exist_ok=True)

    # 6. 处理剩余设备到add组（20个一组）
    print(f"\n开始处理剩余设备（{len(remaining_files)}个设备）")
    group_idx = 1
    for i in range(0, len(remaining_files), ADD_GROUP_SIZE):
        group_files = remaining_files[i:i + ADD_GROUP_SIZE]
        if not group_files:
            break

        add_dir = os.path.join(OUTPUT_DIR, f"add{group_idx}")
        os.makedirs(add_dir, exist_ok=True)
        print(f"\n处理add{group_idx}组（{len(group_files)}个设备）")

        for (file_path, device_id, _) in group_files:
            features = extract_features(file_path)
            if features is None:
                continue

            # 生成所有可能的样本（不限制数量）
            samples = generate_samples(features)
            if not samples:
                print(f"设备 {device_id} 无法生成样本，跳过")
                continue

            # 保存样本
            for j, sample in enumerate(samples):
                save_path = os.path.join(add_dir, f"device_{device_id:02d}_{j:04d}.npy")
                np.save(save_path, sample)

            print(f"设备 {device_id} 处理完成，生成 {len(samples)} 个样本")

        group_idx += 1

    print("\n所有设备处理完成")


# ---------------------------
# 执行处理
# ---------------------------
if __name__ == "__main__":
    process_all_devices()
