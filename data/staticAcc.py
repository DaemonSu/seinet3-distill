import re
import pandas as pd

# 原始数据粘贴在这里（或读取文件）
# 根据参数比较过程中的输出日志，统计相关指标
with open("log.txt", "r") as f:
    raw_data = f.read()

# Step 1: 拆分成多组
blocks = re.split(r"\{[^}]*\}", raw_data)
param_blocks = re.findall(r"\{[^}]*\}", raw_data)

results = []

for param_block, data_block in zip(param_blocks, blocks[1:]):  # skip first empty
    # 提取参数
    param_match = re.search(r"'temperature':\s*([\d.]+).*?'base_margin':\s*([\d.]+).*?'beta':\s*([\d.]+)", param_block)
    if not param_match:
        continue
    temperature, base_margin, beta = map(float, param_match.groups())

    # 提取指标
    cs_match = re.search(r"Closed-set Accuracy\s*:\s*([\d.]+)%", data_block)
    osr_match = re.search(r"Open-set Recognition Rate\s*:\s*([\d.]+)%", data_block)
    oa_match = re.search(r"Overall Accuracy\s*:\s*([\d.]+)%", data_block)
    f1_match = re.search(r"Open-set F1 Score\s*:\s*([\d.]+)", data_block)

    if cs_match and osr_match and oa_match and f1_match:
        closed_set_acc = float(cs_match.group(1))
        open_set_recognition_rate = float(osr_match.group(1))
        overall_acc = float(oa_match.group(1))
        open_set_f1 = float(f1_match.group(1))

        results.append({
            "temperature": temperature,
            "base_margin": base_margin,
            "beta": beta,
            "closed_set_acc": closed_set_acc,
            "open_set_recognition_rate": open_set_recognition_rate,
            "overall_acc": overall_acc,
            "open_set_f1": open_set_f1,
        })

# 保存为 CSV
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print(df)
