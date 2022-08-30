from model import STAR
from trainers import Trainer
from util import *
from config import Config as config
from dataset import TrajectoryDataset

if __name__ == '__main__':
    train_dataset = TrajectoryDataset('./data/eth', config.max_seqlen, config)
    val_dataset = TrajectoryDataset('./data/val',config.max_seqlen,config)
    # model = TrGraphformerModel(config)
    model = STAR(config)
    trainer = Trainer(train_dataset, val_dataset, model, config)
    trainer.train()
