# 准备数据,参考social-stgcnn和trAISformer
import os

import numpy as np
from tqdm import tqdm
from util import *
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """customized pytorch dataset"""

    def __init__(self, data_dir, seq_len, config) -> None:
        super(TrajectoryDataset).__init__()
        # TODO 不知道有没有问题!
        r"""
        Args:
            data_dir::=要读入的已经格式化好了的文件
                <timestep> <ped_id> <x> <y>
            max_seqlen:transformer一个seq的最大长度
                    
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        self.data_dir = data_dir
        self.config = config
        self.max_seqlen = config.max_seqlen
        self.seq_len = seq_len
        # self.data 存储DataFrame对象,一个DataFrame对象表示一个小整体?
        self.data = {'frame_ships': [], 'trajectories': []}
        # self.data_index存放一段一段序列开头的帧id信息
        self.data_index = np.array(
            [np.array([], dtype=int),
             np.array([], dtype=int)])
        # 读取根目录下的所有文件作为测试集或者训练集(如一个eth文件夹为一个整体)
        for seti, path in enumerate(os.listdir(data_dir)):
            print('开始读取{}数据'.format(path))
            # 一个path表示一个独立的整体(但仍然有很多文件)
            path = os.path.join(data_dir, path)
            data = read_epoch(path)

            # 提取帧号
            frames_id = np.unique(data.loc[:, ('timestep')]).tolist()
            # 提取船的mmsi(id)号
            mmsi_list = np.unique(data.loc[:, ('mmsi')]).tolist()
            # 生成每搜船的轨迹信息——时间信息(前提是索引不重复)
            trajectories = {}
            frame_ships = {}
            for _, ship_id in tqdm(enumerate(mmsi_list)):
                # trajectory存放每个人的轨迹信息——时间信息
                trajectory = data[data.loc[:, ('mmsi')] == ship_id]
                # 过短，抛弃(但是不会出现这个情况,以为在预处理时已经抛弃了)
                if len(trajectory) < 2:
                    continue
                # 保存轨迹(trajectories[i]表示id号为i的船的轨迹信息)
                trajectory.reset_index(drop=True, inplace=True)
                # 保存id船的轨迹(trajectories[ship_id]表示第id艘船的轨迹信息)
                trajectories[ship_id] = trajectory

            self.data['trajectories'].append(trajectories)  # 一个文件表示List的一个元素

            # 生成每一帧的行人信息——空间信息
            for _, frame_id in tqdm(enumerate(frames_id)):
                # frame_ship存放id为frame_id的帧的所有行人信息
                frame_ship = data[data.loc[:, ('timestep')] == frame_id]
                frame_ship.reset_index(drop=True, inplace=True)  # 重新排序标号

                # 保存framed_id帧的信息(frame_ship[i]表示i帧的船信息)
                frame_ships[frame_id] = frame_ship

            self.data['frame_ships'].append(frame_ships)
            print('读取{}数据完成'.format(path))
            print('开始准备item数据')

            set_id = []  # 保存frame_id_in_set元素对应的集合的id(即文件对应的id)
            # 保存所有能构成训练序列的帧首(长度即为样本的长度)包含所有文件的——即为了生成start和end
            frame_id_in_set = []
            frames_id = sorted(list(frame_ships.keys()))  # 获取所有的帧号
            # 这里减去seq_length是为了防止最后的序列没办法凑成一个完整的预测序列而越界
            maxframe = max(
                frames_id
            ) - self.seq_len * config.skip  # TODO 这里的skip是我自己加上去的不知道对不对
            frames_id = [x for x in frames_id
                         if not x > maxframe]  #提取出能构成完整序列的帧
            set_id.extend([seti for _ in range(len(frames_id))])
            frame_id_in_set.extend(frames_id)

            self.data_index = np.concatenate([
                self.data_index,
                np.concatenate([
                    np.array([frame_id_in_set], dtype=int),
                    np.array([set_id], dtype=int)
                ], 0)
            ], 1)

            # 打乱data顺序(假的shuffle?每次的epoch都一样的)
            all_frame_list = [i for i in range(len(frames_id))]
            if config.shuffle:
                random.Random().shuffle(all_frame_list)
            self.data_index = self.data_index[:, all_frame_list]

            # TODO 充分利用数据?
            if 'train' == 'train':
                self.data_index = np.append(
                    self.data_index, self.data_index[:, :config.batch_size], 1)

            # TODO 还没划分val和train的数据集(打算在构造batch的时候再划分)

    def __len__(self):
        return self.data_index.shape[1]

    def __getitem__(self, index):
        """Gets items.
        
        Returns:
            nodes:与输入的一样
            num_ships:nodes中船的数量,即节点数量
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        if self.config.split == 'test': skip = 10
        else: skip = 10  # 这里的skip表示相邻两帧的跨越的单位
        cur_frame_id, cur_set = self.data_index[:, index]
        # 保存每个小的单元对应的船数量,因为这里只有一个单元,所以也就是当前index对应的船数量
        num_ships = []
        # TODO 这里出现中间的船怎么办
        # 开始帧的所有船的id
        start_frame_ships = set(
            self.data['frame_ships'][cur_set][cur_frame_id].loc[:, 'mmsi'])
        # 结束帧的所有船的id
        end_frame_ships = set(
            self.data['frame_ships'][cur_set][cur_frame_id +
                                              (self.config.max_seqlen - 1) *
                                              skip].loc[:, 'mmsi'])
        # print(cur_frame_id)
        present_ships = start_frame_ships | end_frame_ships  # 合并、去重,当前区间出现过的所有船
        # if len(start_frame_ships & end_frame_ships) == 0:
        #     return None  # TODO 也包含在后边了?
        # 获取当前区间内每一艘船的轨迹片段
        traject = ()
        for ship in present_ships:
            candidate_traj = self.data['trajectories'][cur_set][ship]
            # cur_traj存放当前ship的当前区间内的所有轨迹——TODO 为什么会断开?
            cur_traj = np.zeros((self.config.max_seqlen, 2))
            end_frame_id = cur_frame_id + self.config.max_seqlen * skip
            # TODO 能不能精简为下面的代码
            # 更新候选序列,需要重新编号,不包括结束帧?(模仿STAR代码)刚好20帧
            candidate_traj = candidate_traj[np.logical_and(
                candidate_traj.loc[:, 'timestep'] >= cur_frame_id,
                candidate_traj.loc[:, 'timestep'] < end_frame_id)].reset_index(
                    drop=True)
            # TODO 连续性判断!应该不会不连续?因为半小时才会间隔，而这不可能超过半小时,所以不需要连续性判断
            # 将区间映射到从0-max_seqlen 上, 不是从0开始!这里转为Int是为了索引需要
            offset_start = int(candidate_traj.loc[0, 'timestep'] -
                               cur_frame_id) // skip
            offset_end = self.config.max_seqlen + int(
                candidate_traj.iloc[-1, 0] - end_frame_id) // skip
            assert offset_end + 1 - offset_start == candidate_traj.shape[0]
            cur_traj[offset_start:offset_end +
                     1, :] = candidate_traj[['x', 'y']]
            # TODO 要不要删去轨迹点较少的船的轨迹?——删了会不会导致船与船间的影响被忽略
            if sum(cur_traj[:, 0] > 0) < 5: continue
            cur_traj = (cur_traj.reshape(-1, 1, 2), )  # 变成三维的信息
            traject = traject.__add__(cur_traj)

        # 当前区间(index对应的data_index区间)内的所有船的信息(时间信息?),合并后同一个时间会合在一起,观看合并的维度变化即可
        traject_batch = np.concatenate(traject, axis=1)
        seq_list, nei_list, nei_num = self.__get_social_inputs_numpy(
            traject_batch)
        num_ships.append(traject_batch.shape[1])

        return traject_batch, seq_list, nei_list, nei_num, num_ships

    def __get_social_inputs_numpy(self, nodes):
        r"""构建图(船是节点,是否相互有影响是边)
        batch:
            shape->[max_seqlen,sum_ships_num_in_frames,2]
            即[最大的序列长度,船在index对应的时间区间内的总的数量,x和y两维]
        return:
            seq_list:
                shape->[max_seqlen,num_ships]
                seq_list[f,m]:第f帧中船m是否出现
            nei_list:
                shape->[max_seqlen,num_ships,num_ships]
                nei_list[f,i,j]:第f帧中船i和船j是否有连接关系
            nei_num:
                shape->[max_seqlen,num_ships]
                nei_num[f,i]:第f帧中和船i有连接关系的船的数量
        """
        num_ships = nodes.shape[1]
        seq_list = np.zeros((nodes.shape[0], num_ships))
        # 这里用range是为了选第二维的数据,遍历每一艘船
        for ship_i in range(num_ships):
            seq = nodes[:, ship_i]
            seq_list[seq[:, 0] != 0, ship_i] = 1  # 1表示在f帧中存在该船

        # 获取相对坐标和相邻节点的id列表
        nei_list = np.zeros((nodes.shape[0], num_ships, num_ships))
        nei_num = np.zeros((nodes.shape[0], num_ships))

        # nei_list[f,i,j]表示在f帧中是否j是i的邻居
        for ship_i in range(num_ships):
            nei_list[:, ship_i, :] = seq_list
            nei_list[:, ship_i, ship_i] = 0  # TODO 自己不是自己的邻居?
            nei_num[:, ship_i] = np.sum(nei_list[:, ship_i, :], 1)
            seq_i = nodes[:, ship_i]
            for ship_j in range(num_ships):
                if ship_j == ship_i: continue  # TODO 避免不必要的计算?
                # 根据距离来删去船之间的联系
                seq_j = nodes[:, ship_j]  # 第j搜船的每一帧的空间信息
                # 选取第i艘船和第j艘船同时出现的帧号
                select_frame_idx = (seq_list[:, ship_i]
                                    == 1) & (seq_list[:, ship_j] == 1)
                # 将预选出来的i和j同时出现的帧号的距离进行计算,剔除距离较远的船之间的联系
                relative_cord = seq_i[select_frame_idx, :] - seq_j[
                    select_frame_idx, :]
                select_distance = (
                    abs(relative_cord[:, 0]) > self.config.neighbor_x_thred
                ) | (abs(relative_cord[:, 1]) > self.config.neighbor_y_thred)
                # 删去与第i艘船距离较大的船之间的边——改变邻接矩阵和邻接节点的数量
                nei_num[select_frame_idx, ship_i] -= select_distance
                select_frame_idx[select_frame_idx == True] = select_distance
                nei_list[select_frame_idx, ship_i, ship_j] = 0

        return seq_list, nei_list, nei_num
