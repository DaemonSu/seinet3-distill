import itertools
import torch
from torch.utils.data import DataLoader

from loss import SupConLoss_DynamicMargin


class GridSearchRunner:
    def __init__(self, train_fn, val_fn, param_grid, device='cuda'):
        """
        Args:
            train_fn: 函数，输入 criterion，返回已训练模型。
            val_fn: 函数，输入模型，返回验证集上的指标（如 accuracy）。
            param_grid: 参数搜索空间，dict，例如：
                {
                    "temperature": [0.05, 0.07, 0.1],
                    "base_margin": [0.0, 0.1, 0.3],
                    "beta": [0.0, 0.2, 0.4],
                }
            device: 训练设备
        """
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.param_grid = param_grid
        self.device = device

    def run(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        best_score = -1
        best_params = None

        results = []

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            print(f"\n[GridSearch] Testing params: {config}")

            # 构造 loss 函数
            criterion = SupConLoss_DynamicMargin(
                temperature=config["temperature"],
                base_margin=config["base_margin"],
                beta=config["beta"]
            ).to(self.device)

            # 训练模型
            model = self.train_fn(criterion)

            # 验证
            score = self.val_fn(model)
            results.append((config, score))

            print(f"[GridSearch] Validation Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = config

        print("\n========== Grid Search Complete ==========")
        print(f"Best Params: {best_params}")
        print(f"Best Validation Score: {best_score:.4f}")
        return best_params, best_score, results
