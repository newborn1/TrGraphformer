# 存放训练所需的各种函数
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

from util import *


# TODO
class Trainer:
    """输入训练或者测试的数据集和模型进行训练和预测"""

    def __init__(self, trajectoryDataset, model, config) -> None:
        self.model = model.cuda()
        self.config = config
        self.loader = DataLoader(trajectoryDataset,
                                 shuffle=True,
                                 pin_memory=True,
                                 batch_size=self.config.batch_size,
                                 num_workers=self.config.num_workers)
        self.__set_optimizer()
        self.writer = SummaryWriter(config.logdir)
        self.writer.add_text('config', str(config))
        self.writer.add_hparams(config)  #不知道行不行

    def __set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def __train_epoch(self, epoch=0):
        """根据条件运行给定方法和次数
        Args:
            epoch:表示训练或预测次数
        
        Return:
        """
        self.model.train()
        loss_epoch = 0

        pbar = tqdm(enumerate(self.loader), total=len(self.loader))
        # 一个batch
        for idx, batch in pbar:
            start = time.time()
            inputs = rotate_shift_batch(inputs, self.config,
                                        self.config.random_ratate)
            inputs = tuple([torch.Tensor(i) for i in batch])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, ship_num = inputs
            # 去掉最后一列,因为数据没有shift做减
            inputs_forward = (batch_abs[:-1], batch_norm[:-1],
                              shift_value[:-1], seq_list[:-1], nei_list[:-1],
                              nei_num[:-1], ship_num)

            self.model.zero_grad()

            outputs = self.model(inputs_forward, iftest=False)

            lossmask, num = get_loss_mask(outputs, seq_list[0], seq_list[1:])
            loss_o = torch.sum(self.criterion(outputs, batch_norm[1:, :, :2]),
                               dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

            self.optimizer.step()

            end = time.time()
            if batch % 10 == 0:
                pbar.set_description(
                    'train-{}/{} (epoch {} - batch_idx {}), train_loss = {:.5f}, time/batch = {:.5f} '
                    .format(batch, len(self.loader), epoch, idx, loss.item(),
                            end - start))

        train_loss_epoch = loss_epoch / len(self.loader)

        return train_loss_epoch

    @torch.no_grad()
    def __test_epoch(self, epoch=0):
        self.model.eval()

        pbar = tqdm(enumerate(self.testloader), total=len(self.testloader))

        for index, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            self.model(seqs)

    def __save_model(self, epoch):
        model_path = '{0}/{1}/{1}_{2}.tar'.format(self.config.save_dir,
                                                  self.config.train_model,
                                                  epoch)
        torch.save(
            {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, model_path)

    def train(self):
        print('开始训练')
        # 采用交叉验证
        val_error, val_final_error = 0, 0
        pbar = tqdm(total_len=self.config.max_epoch)
        for epoch in pbar:
            train_loss = self.__train_epoch(self.config.split, epoch)

            if epoch % self.config.start_val == 0:
                val_error, val_final_error = self.__test_epoch(epoch)

                self.best_ade = val_error if val_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if val_final_error < self.best_fde else self.best_epoch
                self.best_fde = val_final_error if val_final_error < self.best_fde else self.best_fde
                self.__save_model(epoch)

                pbar.set_description(
                    '----epoch {}, train_loss={:.5f}, ADE={:.3f}, FDE={:.3f}, Best_ADE={:.3f}, Best_FDE={:.3f} at Epoch {}'
                    .format(epoch, train_loss, val_error, val_final_error,
                            self.best_ade, self.best_fde, self.best_epoch))
            else:
                pbar.set_description('----epoch {}, train_loss={:.5f}'.format(
                    epoch, train_loss))

            self.writer.add_scalars(
                main_tag='train-val',  # 在同一个图中画多个曲线
                tag_scalar_dict={
                    'train-loss': train_loss,
                    'val-ADE': val_error,
                    'val-FDE': val_final_error
                },
                global_step=epoch)
