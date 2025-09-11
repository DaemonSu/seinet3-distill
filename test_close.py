import torch
from torch.utils.data import DataLoader
from config import parse_args
from dataset import SEIDataset
from util.utils import load_object, accuracy

args = parse_args()
device = args.device
# 加载数据
test_dataset = SEIDataset(args.test_closeData)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)



model = torch.load("model/seinet_49.pth")
model.eval()


prototype = load_object('model/prototype.pkl')
# 测试阶段损失退化到距离损失
# testCriterion = ContrastiveLossWithCE(weight_ce=0)

all_features, all_labels,all_predict = [], [],[]
with torch.no_grad():
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        logits, features = model(x, return_feature=True)
        # 打印所有内容而不省略
        torch.set_printoptions(profile="full")

        # 非零比例（衡量稀疏性）
        non_zero_ratio = (features != 0).float().mean().item()
        print("Non-zero ratio:", non_zero_ratio)


        preds, min_dist,_ = prototype.classify(features)

        all_predict.append(preds)
        all_labels.append(label)

    all_predict = torch.cat(all_predict, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
acc = accuracy(all_predict, all_labels)
print(f'in test Acc: {acc:.4f}')
