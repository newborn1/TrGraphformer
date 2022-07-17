# 这是一个配置文件，可以直接在这里调参

class Config:
    # 预处理
    discrete = 10  # 离散化的间隔大小
    abnormal_dx = 0.02  # x 方向上异常的点的最小值
    abnormal_dy = 0.01  # y 方向上异常的点的最小值
    interpolate_min_point = 10  # 满足插值的最少原始点的个数
    interpolate_min_gap = 30  # 插值时相邻两个插值节点的最小间隔时间(min)

    retrain = True
    max_seqlen = 8
    n_layer = 5
    embedding_size = [32]
    output_size = 2
    dropout_prob = 0.5
    src_mask = None