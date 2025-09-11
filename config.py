import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SEI Training Config')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--old_num_classes', default=8, type=int)
    parser.add_argument('--new_num_classes', default=2, type=int)
    parser.add_argument('--seq_len', default=7000, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--epochs2', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[10, 30, 40], type=float)
    parser.add_argument('--embedding_dim', default=1024, type=int)

    parser.add_argument('--threshold', default=10000, type=int)
    parser.add_argument('--open_threshold', type=float, default=0.92, help='Threshold for open-set test decision')
    parser.add_argument('--open_threshold_train', type=float, default=0.30, help='Threshold for open-set train decision')
    parser.add_argument('--proto_threshold', type=float, default=0.50, help='Threshold for proto open-set decision')

    # 损失中的原型距离损失
    parser.add_argument('--con_weight', default=1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--proto_weight', default=0.5, type=float)


    parser.add_argument('--prototype_momentum', default=0.3, type=float)
    parser.add_argument('--margin', default=0.09, type=float)

    # 蒸馏温度
    parser.add_argument('--distill_temperature', default=2, type=float)
    # 蒸馏损失系数
    parser.add_argument('--alpha_kd', default=1, type=float)
    parser.add_argument('--max_feature_per_class', default=65, type=float)

    parser.add_argument('--incr_batch_size', default=32, type=int)

    # openset 训练集合定义
    parser.add_argument('--train_data_close', default='G:/seidata/32ft-exp2/train2-close', type=str)
    parser.add_argument('--train_data_open', default='G:/seidata/32ft-exp2/train2-open', type=str)

    parser.add_argument('--train_data_new', default='G:/seidata/32ft-exp2/train-add1', type=str)


    # openset 验证集合定义
    parser.add_argument('--val_data', default='G:/seidata/32ft-exp2/val2', type=str)


    # openset 测试集
    parser.add_argument('--test_mixed', default='G:/seidata/32ft-exp2/test2-mixed', type=str)

    # openset 测试集
    parser.add_argument('--test_add1', default='G:/seidata/32ft-exp2/test2-add1', type=str)

    parser.add_argument('--save_dir', default='model/', type=str)

    return parser.parse_args()
