# 假设你已有：
# def train_fn(criterion): return trained_model
# def val_fn(model): return val_accuracy

from util.grid_search_runner import GridSearchRunner

param_grid = {
    "temperature": [0.05, 0.07, 0.1],
    "base_margin": [0.0, 0.1, 0.3],
    "beta": [0.0, 0.2, 0.4],
}

runner = GridSearchRunner(
    train_fn=train_fn,
    val_fn=val_fn,
    param_grid=param_grid,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

best_params, best_score, all_results = runner.run()
