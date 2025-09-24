import os
import numpy as np
import scipy.io as sio
from collections import defaultdict

def process_adsb_data(mat_file_path, output_dir):
    """
    处理ADS-B数据，提取设备消息并保存前100个设备的IQ信号
    生成(2, 总长度)形状数据（2对应I/Q维度，总长度=消息数量×单条消息长度）
    保留原始长度拼接，补充ICAO ID部分清零
    """
    # 1. 加载MAT文件数据
    print("加载数据文件...")
    try:
        mat_data = sio.loadmat(mat_file_path)
    except Exception as e:
        print(f"加载MAT文件失败: {e}")
        return

    # 提取关键数据（匹配MATLAB维度逻辑）
    try:
        icao_lst = mat_data['icaoLst'].flatten()  # 设备ID列表（一维）
        msg_id_lst = mat_data['msgIdLst'].flatten()  # 消息ID列表（用于计算维度）
        raw_i_flat = mat_data['rawIMatrix'].flatten()  # 原始I信号（一维）
        raw_q_flat = mat_data['rawQMatrix'].flatten()  # 原始Q信号（一维）
    except KeyError as e:
        print(f"MAT文件中缺少关键数据字段: {e}")
        return

    # 2. 复现MATLAB的reshape操作，转换为二维矩阵
    # MATLAB逻辑：rawIMatrix = reshape(rawIMatrix', len(rawIMatrix)/len(msgIdLst), len(msgIdLst))'
    num_msg = len(msg_id_lst)  # 消息总数量
    if len(raw_i_flat) % num_msg != 0 or len(raw_q_flat) % num_msg != 0:
        print("错误：原始I/Q信号长度不能被消息数量整除，无法完成reshape")
        return
    signal_len = len(raw_i_flat) // num_msg  # 单条消息的原始长度（固定值）

    # 重塑并转置（结果为 (num_msg, signal_len)）
    raw_i_matrix = raw_i_flat.reshape(signal_len, num_msg).T  # I信号矩阵 (总消息数, 单条长度)
    raw_q_matrix = raw_q_flat.reshape(signal_len, num_msg).T  # Q信号矩阵 (总消息数, 单条长度)

    # 3. 统计每个设备的消息数量
    print("统计设备消息数量...")
    device_counts = defaultdict(int)
    for icao in icao_lst:
        device_counts[icao] += 1

    if not device_counts:
        print("错误：未找到任何设备数据")
        return
    # 按消息数量排序并筛选前100个设备
    sorted_devices = sorted(device_counts.items(), key=lambda x: x[1], reverse=True)
    top100_devices = [device for device, _ in sorted_devices[:100]]
    print(f"筛选出消息数量最多的100个设备，最多消息数: {sorted_devices[0][1]}")

    # 4. 处理并保存每个设备的(2, 总长度)形状IQ信号
    print("开始保存IQ信号数据...")
    os.makedirs(output_dir, exist_ok=True)
    for idx, target_icao in enumerate(top100_devices, 0):
        # 找到当前设备的所有数据索引
        device_indices = np.where(icao_lst == target_icao)[0]
        if len(device_indices) == 0:
            print(f"警告：设备 {target_icao} 未找到数据，跳过")
            continue

        # 提取对应设备的I/Q信号（形状：(消息数量, 单条长度)）
        i_signals = raw_i_matrix[device_indices, :]  # (N条消息, signal_len)
        q_signals = raw_q_matrix[device_indices, :]  # (N条消息, signal_len)

        # 清零前256列（ICAO ID部分，与MATLAB逻辑对齐）
        if signal_len >= 256:
            i_signals[:, :256] = 0
            q_signals[:, :256] = 0

        # 核心修改：拼接所有消息，形成(2, 总长度)
        # 总长度 = 消息数量 × 单条消息长度
        total_length = i_signals.shape[0] * i_signals.shape[1]
        # 拼接I信号：将(N, L)展平为(1, N×L)；Q信号同理
        i_combined = i_signals.flatten().reshape(1, -1)  # (1, 总长度)
        q_combined = q_signals.flatten().reshape(1, -1)  # (1, 总长度)
        # 堆叠为(2, 总长度)：第一行I，第二行Q
        iq_combined = np.vstack([i_combined, q_combined])  # (2, 总长度)

        # 保存为.npy文件
        output_path = f"{output_dir}/device_{int(idx):02d}_iq.npy"
        try:
            np.save(output_path, iq_combined)
            print(f"设备 {int(target_icao)} 拼接完成，形状: {iq_combined.shape}")
        except Exception as e:
            print(f"保存设备 {int(target_icao)} 数据失败: {e}")
            continue

        if idx % 10 == 0:
            print(f"已完成{idx}/100个设备数据处理")

    print("所有设备数据处理完成!")


if __name__ == "__main__":
    # 配置文件路径（根据实际情况修改）
    MAT_FILE_PATH = "G:/seidata/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning/adsb_bladerf2_10M_qt0.mat"  # 输入MAT文件路径
    OUTPUT_DIRECTORY = "G:/seidataforCIL/IQArray"  # 输出目录

    # 执行处理流程
    process_adsb_data(MAT_FILE_PATH, OUTPUT_DIRECTORY)
