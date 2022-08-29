from model import STAR
from trainers import Trainer
from util import *
from config import Config as config
from dataset import TrajectoryDataset

if __name__ == '__main__':
    dataset = TrajectoryDataset('./data/eth', config.max_seqlen, config)
    # model = TrGraphformerModel(config)
    model = STAR(config)
    trainer = Trainer(dataset, model, config)
    trainer.train()
