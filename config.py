import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SEI Training Config')
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--old_num_classes', default=20, type=int)
    parser.add_argument('--new_num_classes', default=20, type=int)
    parser.add_argument('--seq_len', default=3000, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=260, type=int)

    parser.add_argument('--batch_size', default=320, type=int)
    # 初始训练过程中，每轮训练开集数据的数量
    parser.add_argument('--open_batch_size', default=80, type=int)

    # 增量学习过程中的每轮数量
    parser.add_argument('--incr_batch_size', default=100, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=0.1, type=float)

    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_epochs', default=[120, 180, 210], type=float)
    parser.add_argument('--embedding_dim', default=1024, type=int)

    # 增量训练过程中的学习率参数设置
    parser.add_argument('--epochs2', default=150, type=int)

    parser.add_argument('--incre_lr', default=0.01, type=float)

    parser.add_argument('--incre_lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--incre_lr_decay_epochs', default=[50, 80, 140], type=float)




    parser.add_argument('--threshold', default=10000, type=int)
    parser.add_argument('--open_threshold', type=float, default=0.50, help='Threshold for open-set test decision')
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
    parser.add_argument('--alpha_kd', default=2, type=float)
    parser.add_argument('--max_feature_per_class', default=80, type=float)



    # openset 训练集合定义
    parser.add_argument('--train_data_close', default='G:/seidataforCIL2/train-closed', type=str)
    parser.add_argument('--train_data_open', default='G:/seidataforCIL2/train-openset', type=str)

    # 初始训练后的闭集测试集
    parser.add_argument('--test_closed', default='G:/seidataforCIL2/test-closed', type=str)
    # 初始训练后的开集测试
    parser.add_argument('--test_mixed', default='G:/seidataforCIL2/test-mixed', type=str)

    #第一次增量学习数据与测试
    parser.add_argument('--train_data_add1', default='G:/seidataforCIL/train-add1', type=str)

    parser.add_argument('--test_add1', default='G:/seidataforCIL/test-add1', type=str)

    # 第2次增量学习数据与测试
    parser.add_argument('--train_data_add2', default='G:/seidataforCIL/train-add2', type=str)

    parser.add_argument('--test_add2', default='G:/seidataforCIL/test-add2', type=str)

    # 第3次增量学习数据与测试
    parser.add_argument('--train_data_add3', default='G:/seidataforCIL/train-add3', type=str)

    parser.add_argument('--test_add3', default='G:/seidataforCIL/test-add3', type=str)







    parser.add_argument('--save_dir', default='model/', type=str)

    return parser.parse_args()
