import os
import torch
import torch.nn as nn
from torchviz import make_dot
import hiddenlayer as hl
import netron
import tempfile
from typing import Optional, List, Dict, Any, Union

from config import parse_args
from model_open import FeatureExtractor


# 全面修复 HiddenLayer 与新版本 PyTorch 的兼容性问题
def patch_hiddenlayer():
    """修补 HiddenLayer 以兼容 PyTorch 1.10+"""
    try:
        # 检查是否需要修补
        if not hasattr(hl.graph, '_optimize_trace'):
            return

        # 修补 _optimize_trace 方法
        import onnx
        from torch.onnx import utils

        def _optimize_trace(trace, operator_export_type):
            graph = trace.graph()
            graph = utils._optimize_graph(graph, operator_export_type, torch.onnx.OperatorExportTypes.ONNX)
            return graph

        hl.graph._optimize_trace = _optimize_trace

        # 修补 graphviz 渲染方法以避免 IPython 依赖
        def save(self, path, format=None):
            """保存图形为文件"""
            if not format:
                format = path.split(".")[-1]
            dot = self.build_dot()
            dot.render(path, format=format, cleanup=True, view=False)
            return path

        hl.graph.Graph.save = save

        print("HiddenLayer 已成功修补以兼容当前 PyTorch 版本")

    except Exception as e:
        print(f"修补 HiddenLayer 失败: {e}")


# 应用修补
patch_hiddenlayer()


class ModelVisualizer:
    """
    模型可视化工具类，支持多种可视化方法
    """

    def __init__(self, model: nn.Module, input_size: tuple, batch_size: int = 1):
        """
        初始化模型可视化工具

        Args:
            model: PyTorch 模型
            input_size: 输入张量的尺寸，不包含批次维度，例如 (3, 224, 224)
            batch_size: 批次大小，默认为 1
        """
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device(
            'cpu')

        # 创建示例输入张量
        self.dummy_input = self._create_dummy_input()

    def _create_dummy_input(self) -> torch.Tensor:
        """创建示例输入张量"""
        # 添加批次维度
        input_shape = (self.batch_size,) + self.input_size
        return torch.randn(input_shape).to(self.device)

    def visualize_with_graphviz(self, output_path: str = "model_graph", format: str = "png") -> None:
        """
        使用 torchviz 和 Graphviz 可视化模型

        Args:
            output_path: 输出文件路径（不含扩展名）
            format: 输出格式，如 'png', 'pdf', 'svg' 等
        """
        try:
            # 确保Graphviz可执行文件在PATH中
            import graphviz
            graphviz.version()
        except Exception as e:
            print(f"Graphviz 未正确安装: {e}")
            print("请确保已安装 Graphviz 软件并将其添加到系统PATH中")
            return

        # 设置中文字体支持
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

        # 生成计算图
        output = self.model(self.dummy_input)
        dot = make_dot(output, params=dict(self.model.named_parameters()))

        # 设置图形属性
        dot.attr(rankdir='TB', size='20,20', dpi='300')
        dot.attr('node', shape='box', style='filled', color='lightblue')

        # 渲染图形
        dot.render(output_path, format=format, cleanup=True, view=False)
        print(f"模型已保存为: {output_path}.{format}")

    def visualize_with_hiddenlayer(self, output_path: str = "model_hiddenlayer", format: str = "png",
                                   simplify: bool = True) -> None:
        """
        使用 HiddenLayer 库可视化模型

        Args:
            output_path: 输出文件路径（不含扩展名）
            format: 输出格式，如 'png', 'pdf', 'svg' 等
            simplify: 是否简化图形，合并常见操作
        """
        try:
            # 创建转换规则以简化图形
            transforms = []
            if simplify:
                transforms = [
                    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
                    hl.transforms.Fold("Linear > Relu", "LinearRelu"),
                    hl.transforms.Fold("Dropout > Linear > Relu", "DropoutLinearRelu"),
                ]

            # 构建计算图
            graph = hl.build_graph(self.model, self.dummy_input, transforms=transforms)

            # 设置主题
            graph.theme = hl.graph.THEMES["blue"].copy()

            # 保存图形（强制使用修补后的方法）
            graph.save(output_path, format=format)
            print(f"模型已保存为: {output_path}.{format}")

        except Exception as e:
            print(f"使用 HiddenLayer 可视化失败: {e}")
            print("提示: 可以尝试使用 visualize_with_torchview 方法替代")

    def export_to_onnx(self, output_path: str = "model.onnx", open_netron: bool = True) -> None:
        """
        将模型导出为 ONNX 格式，并可选择使用 Netron 打开

        Args:
            output_path: 输出 ONNX 文件路径
            open_netron: 是否自动打开 Netron 查看模型
        """
        try:
            # 检查 ONNX 是否已安装
            import onnx
            import onnxruntime

            # 导出模型为 ONNX 格式
            torch.onnx.export(
                self.model,
                self.dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            # 验证导出的模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            print(f"模型已成功导出为: {output_path}")

            # 打开 Netron
            if open_netron:
                try:
                    netron.start(output_path)
                    print("Netron 已启动，正在浏览器中显示模型...")
                    print("若未自动打开，请手动在浏览器中访问: http://localhost:8080")
                except Exception as e:
                    print(f"无法自动启动 Netron: {e}")
                    print(f"请手动打开 Netron 并加载文件: {os.path.abspath(output_path)}")

        except ImportError:
            print("请安装 onnx 和 onnxruntime: pip install onnx onnxruntime")
        except Exception as e:
            print(f"导出 ONNX 模型失败: {e}")

    def visualize_layers(self, output_path: str = "model_layers", format: str = "png") -> None:
        """
        可视化模型的层结构（简化版）

        Args:
            output_path: 输出文件路径（不含扩展名）
            format: 输出格式
        """
        try:
            from graphviz import Digraph

            # 创建有向图
            dot = Digraph(comment='Model Layers')
            dot.attr(rankdir='TB', size='12,12')

            # 记录已添加的节点
            added_nodes = set()

            # 添加输入节点
            input_node = f"input\n{self.dummy_input.shape}"
            dot.node("input", label=input_node, shape='box', style='filled', color='lightgreen')
            added_nodes.add("input")

            prev_node = "input"

            # 遍历模型的子模块
            for i, (name, module) in enumerate(self.model.named_children()):
                try:
                    # 计算模块输出
                    if i == 0:
                        output = module(self.dummy_input)
                    else:
                        output = module(output)

                    # 创建模块节点
                    module_node = f"module_{i}"
                    module_label = f"{name}\n{module.__class__.__name__}"
                    dot.node(module_node, label=module_label, shape='box', style='filled', color='lightblue')
                    added_nodes.add(module_node)

                    # 创建输出节点
                    output_node = f"output_{i}"
                    output_label = f"Output\n{output.shape}"
                    dot.node(output_node, label=output_label, shape='ellipse', style='filled', color='lightgreen')
                    added_nodes.add(output_node)

                    # 添加边
                    dot.edge(prev_node, module_node)
                    dot.edge(module_node, output_node)

                    prev_node = output_node

                except Exception as e:
                    print(f"处理模块 {name} 时出错: {e}")
                    continue

            # 渲染图形
            dot.render(output_path, format=format, cleanup=True, view=False)
            print(f"模型层结构已保存为: {output_path}.{format}")

        except Exception as e:
            print(f"可视化层结构失败: {e}")


# 使用示例
if __name__ == "__main__":
    config = parse_args()
    model = FeatureExtractor(1024).to(config.device)
    # 创建可视化工具
    visualizer = ModelVisualizer(
        model=model,
        input_size=(7000,3)  # 输入尺寸 (C, H, W)
    )

    # 可视化模型
    visualizer.visualize_with_graphviz(output_path="simple_model_graphviz")
    visualizer.visualize_with_hiddenlayer(output_path="simple_model_hiddenlayer")
    visualizer.export_to_onnx(output_path="simple_model.onnx")
    visualizer.visualize_layers(output_path="simple_model_layers")
