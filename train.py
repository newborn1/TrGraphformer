from util import *
from config import Config as config
from dataset import TrajectoryDataset

if __name__ == '__main__':
    datasets = TrajectoryDataset('./data/eth', config.max_seqlen)
