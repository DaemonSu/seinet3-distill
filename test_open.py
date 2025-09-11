import torch
from torch.utils.data import DataLoader
from config import parse_args
from dataset import SEIDataset
from util.utils import load_object

args = parse_args()
device = args.device
# 加载数据
test_dataset = SEIDataset(args.test_data)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



model = torch.load("model/seinet_49.pth")
model.eval()

model_second = torch.load("model/train_second4.pth")
model_second.eval()



prototype = load_object('model/prototype.pkl')

all_features, all_labels,all_predict = [], [],[]
with torch.no_grad():
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        logits, feature = model(x, return_feature=True)
        preds, min_dist,diff_dist = prototype.classify(feature)

        min_dist = min_dist.unsqueeze(1)  # 扩展维度为 [64, 1]
        # x3 = torch.cat([min_dist,diff_dist], dim=1)  # 拼接特征

        label = label.float()
        binary_labels = (label < 10).float().unsqueeze(1)  # 小于10为已知 → 1，否则未知 → 0
        isopen = model_second(diff_dist)

        probs = torch.sigmoid(isopen)
        predictions = (probs >= 0.5).float()


        # 将概率值与 0.5 比较，大于等于 0.5 认为是未知设备，否则是已知设备
        # predictions = (isopen >= 500).float()

        all_predict.append(predictions)
        all_labels.append(label)



  # 拼接为张量
    all_predict = torch.cat(all_predict, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 转成二元标签
    binary_labels = (all_labels < 10).float().unsqueeze(1)

    # 计算开集闭集准确率
    correct = (all_predict == binary_labels).float().sum()
    total = binary_labels.numel()
    accuracy = correct / total

    print(f'测试集数据总量为 {total}，Acc: {accuracy:.4f}')
