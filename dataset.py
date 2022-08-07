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
        # self.sample_data存放一段一段序列开头的帧id信息
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
            # 这里减去seq_length是为了防止最后的序列没办法凑成一个完整的预测序列
            maxframe = max(frames_id) - self.seq_len
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

            # 充分利用数据
            if 'train' == 'train':
                self.data_index = np.append(
                    self.data_index, self.data_index[:, :config.batch_size], 1)

            # TODO 还没划分val和train的数据集(打算在构造batch的时候再划分)

    def __getitem__(self, index):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        if self.config.split == 'test': skip = 1
        else: skip = 1  # 这里的skip表示相邻两帧的跨越的单位
        cur_frame_id, cur_set = self.data_index[:, index]
        # 开始帧的所有船的id
        start_frame_ships = set(
            self.data['frame_ships'][cur_set][cur_frame_id].loc[:, 'mmsi'])
        # 结束帧的所有船的id
        end_frame_ships = set(
            self.data['frame_ships'][cur_set][cur_frame_id +
                                              self.config.max_seqlen *
                                              skip].loc[:, 'mmsi'])
        present_ships = start_frame_ships | end_frame_ships  # 合并、去重,当前区间出现过的所有船
        if len(start_frame_ships & end_frame_ships) == 0:
            return None  # TODO 抛弃轨迹?index不对应怎么办
        # 获取当前区间内每一艘船的轨迹片段
        traject = ()
        for ship in present_ships:
            candidate_traj = self.data['trajectories'][cur_set][ship]
            cur_traj = np.zeros((self.config.max_seqlen, 3))
            start_n = np.where(candidate_traj.loc[:,
                                                  'timestep'] == cur_frame_id)
            end_n = np.where(candidate_traj.loc[:,
                                                'timestep'] == cur_frame_id +
                             skip * self.config.max_seqlen)
            # TODO 能不能精简为下面的条件
            if start_n[0].shape[0] == 0 or end_n[0].shape[0] == 0:  # 有一个没有查到
                start_n = 0
                end_n = 0
            else:
                end_n = end_n[0][0]
                start_n = start_n[0][0]

            # 更新候选序列
            candidate_traj = candidate_traj[start_n:end_n, :]
            # 将区间映射到从0开始
            offset_end = len(candidate_traj)
            cur_traj[:offset_end, :] = candidate_traj.loc[:, ('x', 'y')]
            # TODO 要不要删去轨迹点较少的船的轨迹?——删了会不会导致船与船间的影响被忽略
            if len(cur_traj[:, 0] > 0) < 5: continue
            cur_traj = (cur_traj.reshape(-1, 1, 2), )  # 变成三维的信息
            traject = traject.__add__(cur_traj)

        # 当前区间(index对应的data_index区间)内的所有船的信息(时间信息)
        traject_batch = np.concatenate(traject, axis=1)
        seq = self.__get_seq(index)
        mask = self.__get_mask(index)
        seqlen = self.__get_seqlen(index)
        mmsi = self.__get_mmsi(index)

        return traject_batch

    def __len__(self):
        return self.data_index.shape[1]

    def __get_seq(self, index):
        pass

    def __get_mask(self, index):
        pass

    def __get_seqlen(index):
        pass

    def __get_mmsi(index):
        pass
