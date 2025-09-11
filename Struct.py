from graphviz import Digraph

def visualize_fftblock_fix_garbled():
    dot = Digraph(
        name="FFTBlock",
        format="pdf",
        graph_attr={
            "rankdir": "TB",
            "size": "12,8",
            "fontpath": "C:/Windows/Fonts/",  # 强制字体路径（Windows）
            "fontname": "Microsoft YaHei, Courier",  # 中+英字体组合
            "dpi": "300"
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontpath": "C:/Windows/Fonts/",  # 节点字体路径
            "fontname": "Microsoft YaHei, Courier",  # 中+英字体
            "fontsize": "10",
            "penwidth": "1.5"
        },
        edge_attr={
            "penwidth": "1.2",
            "color": "#333333"
        }
    )

    # 节点标签全部改为纯英文（若不需要中文，可彻底避免编码问题）
    dot.node("input", label="Input\n[B, C, T] (Temporal Features)", **{"fillcolor": "#E6F3FF"})
    dot.node("fft_submodule", label=(
        "RFFT + Magnitude\n"
        "Ops: Real FFT on Time Dim → Abs\n"
        "Code: torch.fft.rfft(...) → .abs()\n"
        "Output: [B, C, (T//2)+1] (Frequency Bins)"
    ), **{"fillcolor": "#F5F5F5", "shape": "Mrecord"})
    dot.node("interpolate", label=(
        "Linear Interpolation\n"
        "Code: F.interpolate(mode='linear')\n"
        "Output: [B, C, T] (Restore Time Steps)"
    ), **{"fillcolor": "#FFFFFF"})
    dot.node("conv1d", label=(
        "Conv1d\n"
        "Config: in=C, out=C, kernel=1\n"
        "Code: self.conv(fft_mag)\n"
        "Output: [B, C, T]"
    ), **{"fillcolor": "#FFFFFF"})
    dot.node("bn", label=(
        "BatchNorm1d\n"
        "Ops: Normalize Along Channel Dim\n"
        "Code: self.bn(out)\n"
        "Output: [B, C, T]"
    ), **{"fillcolor": "#FFFFFF"})
    dot.node("relu", label=(
        "ReLU\n"
        "Code: self.relu(...)\n"
        "Output: [B, C, T]"
    ), **{"fillcolor": "#FFFFFF"})
    dot.node("output", label="Output\n[B, C, T] (Processed Features)", **{"fillcolor": "#E6F3FF"})

    # 连接节点
    dot.edges([
        ("input", "fft_submodule"),
        ("fft_submodule", "interpolate"),
        ("interpolate", "conv1d"),
        ("conv1d", "bn"),
        ("bn", "relu"),
        ("relu", "output")
    ])

    # 保存（指定渲染引擎为 dot，确保字体嵌入）
    dot.render("FFTBlock_Fixed", directory="./", view=True, engine="dot")
    print("修正乱码后的结构图已保存为 FFTBlock_Fixed.pdf")

# 运行
visualize_fftblock_fix_garbled()
