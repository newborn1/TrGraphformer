class Config:
    """这是一个配置文件(存放需要频繁调参的参数)，可以直接在这里调参"""
    # 预处理(离散化数据、对齐时间、插值补齐数据、去除不合适轨迹)
    discrete = 10  # 离散化的间隔大小，即skip
    abnormal_dx = 0.02  # x 方向上异常的点的最小值
    abnormal_dy = 0.01  # y 方向上异常的点的最小值
    interpolate_min_point = 10  # 满足插值的最少原始点的个数
    interpolate_min_gap = 30  # 插值时相邻两个插值节点的最小间隔时间(min)
    pre_show = True  # 可视化预处理后的数据

    # 准备数据(生成TrajectoryDataset)
    max_seqlen = 128  # 这个还需要调整
    skip = discrete
    neighbor_x_thred = 0.04  # 相邻最短距离,TODO 调整,2km——20min
    neighbor_y_thred = 0.02

    # 保存文件
    save_dir = './output/eth'
    save_base_dir = './output'
    logdir = './output/log/run'
    train_model = 'trGraphformer'
    clip = 1  # 用在trainers中的__train_epoch
    # 训练
    random_rotate = True
    split = 'retrain'  # (有retrain和test)
    num_workers = 0  # 给DataLoader配置的
    batch_size = 1  # 32
    max_epoch = 64 + 1  # 训练的批次
    start_val = 8  # 8个epoch开始评估、保存一次
    learning_rate = 0.0015
    obs_len = 8
    pred_len = 12
    max_seqlen = obs_len + pred_len
    n_layer = 5
    embedding_size = [32]
    output_size = 2
    dropout_prob = 0.5
    src_mask = None