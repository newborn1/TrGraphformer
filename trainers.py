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

    def _init_(self, train_trajectoryDataset, test_trajectoryDataset, model, config) -> None:
        self.model = model.cuda()
        self.config = config
        self.trainloader = DataLoader(train_trajectoryDataset,
                                 shuffle=True,
                                 pin_memory=True,
                                 batch_size=self.config.batch_size,
                                 num_workers=self.config.num_workers)
        self.valloader = DataLoader(test_trajectoryDataset,
                                     shuffle=True,
                                     pin_memory=True,
                                     batch_size=1,
                                     num_workers=self.config.num_workers)
        self._set_optimizer()
        self.writer = SummaryWriter(config.logdir)

    def _set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def _train_epoch(self, epoch=0):
        """根据条件运行给定方法和次数
        Args:
            epoch:表示训练或预测次数
        
        Return:
        """
        self.model.train()
        loss_epoch = 0

        pbar = tqdm(enumerate(self.trainloader), total=len(self.trainloader))
        # 一个batch
        for idx, batch in pbar:
            start = time.time()
            # TODO 目前只支持一个batch
            batch = [batch[i][0] for i in range(len(batch))]
            inputs = rotate_shift_batch(batch, self.config,
                                        self.config.random_rotate)
            inputs = tuple([i.float().cuda() for i in inputs])

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

            loss = (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

            self.optimizer.step()

            end = time.time()
            if idx % 1 == 0:
                pbar.set_description(
                    'train-(epoch {} - batch_idx {}), train_loss = {:.5f}, time/batch = {:.5f} '
                    .format(epoch, idx, loss.item(), end - start))

        train_loss_epoch = loss_epoch / len(self.trainloader)

        return train_loss_epoch

    def test(self):
        print('开始测试')
        self.model.eval()
        test_error, test_final_error = self._test_epoch()
        print(f'test_error: {test_error} test_final_error: {test_final_error}')

    @torch.no_grad()
    def _test_epoch(self, epoch=0):
        # TODO 还未完成(返回值)
        self.model.eval()
        error_epoch,final_error_epoch = 0,0
        error_cnt_epoch,final_error_cnt_epoch = 1e-5,1e-5

        pbar = tqdm(enumerate(self.valloader), total=len(self.valloader))
        # 一个batch
        for idx, batch in pbar:
            start = time.time()
            # TODO 目前只支持一个batch
            batch = [batch[i][0] for i in range(len(batch))]
            inputs = tuple([i.float().cuda() for i in inputs])

            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, ship_num = inputs
            # 去掉最后一列,因为数据没有shift做减
            inputs_forward = (batch_abs[:-1], batch_norm[:-1],
                              shift_value[:-1], seq_list[:-1], nei_list[:-1],
                              nei_num[:-1], ship_num)

            outputs = self.model(inputs_forward, iftest=True)
            targets = batch_norm[1:, :, :2]
            error = torch.norm(outputs - targets, p=2, dim=3)
            error_epoch += error
            final_error_epoch += error

            end = time.time()
            if idx % 1 == 0:
                pbar.set_description(
                    'train-(epoch {} - batch_idx {}), val_error = {:.5f}, time/batch = {:.5f} '
                        .format(epoch, idx, error, end - start))

        val_error_epoch = error_epoch / len(self.valloader)
        val_final_error_epoch = final_error_epoch / len(self.valloader)

        return val_error_epoch, val_final_error_epoch


    def _save_model(self, epoch):
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
        pbar = tqdm(range(self.config.max_seqlen))
        for epoch in pbar:
            train_loss = self._train_epoch(epoch)

            if epoch % self.config.start_val == 0:
                val_error, val_final_error = self._test_epoch(epoch)

                self.best_ade = val_error if val_final_error < self.best_fde else self.best_ade
                self.best_epoch = epoch if val_final_error < self.best_fde else self.best_epoch
                self.best_fde = val_final_error if val_final_error < self.best_fde else self.best_fde
                self._save_model(epoch)

                self.writer.add_hparams(
                    {
                        'batchsize': self.config.batch_size,
                        'lr': self.config.learning_rate,
                        'epoch': epoch
                    }, {
                        'accuracy': val_error,
                        'loss': train_loss
                    })

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
