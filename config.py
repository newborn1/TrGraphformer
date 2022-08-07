class Config:
    """这是一个配置文件(存放需要频繁调参的参数)，可以直接在这里调参"""
    # 预处理(离散化数据、对齐时间、插值补齐数据、去除不合适轨迹)
    discrete = 10  # 离散化的间隔大小
    abnormal_dx = 0.02  # x 方向上异常的点的最小值
    abnormal_dy = 0.01  # y 方向上异常的点的最小值
    interpolate_min_point = 10  # 满足插值的最少原始点的个数
    interpolate_min_gap = 30  # 插值时相邻两个插值节点的最小间隔时间(min)
    pre_show = True  # 可视化预处理后的数据

    # 准备数据(生成TrajectoryDataset)
    max_seqlen = 128  # 这个还需要调整

    # 训练
    split = 'retrain'  # (有retrain和test)
    num_works = 0  # 给DataLoader配置的
    batch_size = 32
    max_epoch = 256  # 训练的批次
    max_seqlen = 20
    n_layer = 5
    embedding_size = [32]
    output_size = 2
    dropout_prob = 0.5
    src_mask = None