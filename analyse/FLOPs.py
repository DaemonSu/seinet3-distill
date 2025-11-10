import os
import torch
import torch.nn as nn
import time
from thop import profile
from thop import clever_format

# =============================
# 依赖说明：
# pip install thop
# =============================

def measure_latency(model, input_shape=(1, 3, 680), device='cuda', n_warmup=10, n_iter=100):
    """
    测量推理延迟 (毫秒)，使用CUDA事件确保准确。
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # 正式计时
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(n_iter):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # ms
            timings.append(curr_time)

    mean_time = sum(timings) / len(timings)
    std_time = (sum((t - mean_time) ** 2 for t in timings) / len(timings)) ** 0.5
    return mean_time, std_time
def infer_input_shape(encoder):
    name = encoder.__class__.__name__.lower()
    if "1d" in name:
        return (1, 3, 680)
    elif "2d" in name or "mobilenet" in name:
        return (1, 3, 32, 32)
    else:
        return (1, 3, 680)


def analyse_model(encoder, projector, classifier, input_shape=(1, 3, 680), device='cuda'):
    """
    分析模型参数量、FLOPs、大小与推理延迟。
    encoder, projector, classifier 需为 nn.Module 实例
    """
    print("========== Model Complexity Analysis ==========")
    model = nn.Sequential(encoder, projector, classifier).to(device)
    model.eval()

    # 计算 FLOPs 与参数量
    dummy_input = torch.randn(*input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    # 模型大小 (MB)
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / (1024 ** 2)
    os.remove("temp_model.pth")

    # 测量推理延迟
    mean_time, std_time = measure_latency(model, input_shape, device=device)
    fps = 1000.0 / mean_time

    print(f"Parameters: {params}")
    print(f"FLOPs: {flops}")
    print(f"Model Size: {size_mb:.2f} MB")
    print(f"Latency: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("===============================================")

    return {
        "Params": params,
        "FLOPs": flops,
        "Size(MB)": size_mb,
        "Latency(ms)": mean_time,
        "Latency_std(ms)": std_time,
        "FPS": fps
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ 加载你保存的模型组件（根据增量框架命名规则）
    encoder = torch.load("../model/encoder_step10.pth", map_location=device)
    projector = torch.load("../model/contrastive_step10.pth", map_location=device)
    classifier = torch.load("../model/classifier_step10.pth", map_location=device)

    # ✅ 设定输入维度：根据你的模型输入 [B, C, L]，C=3, L=680
    input_shape = (1, 29, 42, 3)
    results = analyse_model(encoder, projector, classifier, input_shape=input_shape, device=device)

    # results = analyse_model(
    #     encoder,
    #     projector,
    #     classifier,
    #     input_shape=(1, 3, 680),
    #     device=device
    # )

    print("\n[✓] Complexity summary ready for paper reporting.")
