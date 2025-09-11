import os
import shutil
import random
from collections import defaultdict

# 每个设备抽取的样本数量
N_PER_DEVICE = 700

# # 训练开集源目录列表
source_dirs = ['G:/seidata/2ft-exp/train','G:/seidata/8ft-exp/train','G:/seidata/14ft-exp/train','G:/seidata/20ft-exp/train','G:/seidata/26ft-exp/train']
#
# # 扁平目标目录
target_dir = 'G:/seidata/allft-exp/train2-close'

#训练闭集 源目录列表
# source_dirs = ['G:/seidata/2ft-exp/val','G:/seidata/8ft-exp/val','G:/seidata/14ft-exp/val','G:/seidata/20ft-exp/val','G:/seidata/26ft-exp/val']
#
# # 扁平目标目录
# target_dir = 'G:/seidata/allft-exp/train2-open'


os.makedirs(target_dir, exist_ok=True)

for source in source_dirs:
    # 正确提取距离标签，如 "2ft"
    exp_folder = os.path.basename(os.path.dirname(source))  # 得到 '2ft-exp'
    distance_label = exp_folder.split('-')[0]               # 得到 '2ft'

    # 收集每个设备的文件列表
    device_files = defaultdict(list)
    for filename in os.listdir(source):
        if not filename.endswith('.npy'):
            continue
        parts = filename.replace('.npy', '').split('_')
        if len(parts) != 3:
            continue
        device_id = parts[1]
        device_files[device_id].append(filename)

    # 对每个设备抽样
    for device_id, file_list in device_files.items():
        if len(file_list) < N_PER_DEVICE:
            print(f"[警告] {source} 中 device_{device_id} 样本不足（仅有 {len(file_list)} 条），全部拷贝")
            sampled = file_list
        else:
            sampled = random.sample(file_list, N_PER_DEVICE)

        # 拷贝并重命名
        for fname in sampled:
            data_id = fname.replace('.npy', '').split('_')[2]
            new_fname = f"device_{device_id}_{data_id}_{distance_label}.npy"
            src_path = os.path.join(source, fname)
            dst_path = os.path.join(target_dir, new_fname)
            # 拷贝文件（你也可以用 shutil.move() 进行“移动”）

            # shutil.copy2(src_path, dst_path)
            shutil.move(src_path, dst_path)
