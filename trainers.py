# 存放训练所需的各种函数
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


# TODO
class Trainer:
    """输入训练或者测试的数据集和模型进行训练和预测"""

    def __init__(self, trajectoryDataset, model, config) -> None:
        self.model = model
        self.dataset = trajectoryDataset
        self.config = config

    def __run_epoch(self, split='retrain', epoch=0):
        """根据条件运行给定方法和次数
        Args:
            split:用于判断是训练还是预测,因为他们需要的mask等操作不一样
            epoch:表示训练或预测次数
        
        Return:
        """
        loader = DataLoader(self.dataset,
                            shuffle=True,
                            pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        pbar = tqdm(enumerate(loader), total=len(loader))

        for index, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            self.model(seqs, masks)

    def train(self, config):
        for epoch in range(config.max_epoch):
            self.__run_epoch(config.split, epoch)
