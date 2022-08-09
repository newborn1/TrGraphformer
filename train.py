from util import *
from tqdm import tqdm
from config import Config as config
from dataset import TrajectoryDataset

if __name__ == '__main__':
    datasets = TrajectoryDataset('./data/eth', config.max_seqlen, config)
    data_len = len(datasets)
    for idx, batch in enumerate(datasets):
        if idx % 50 == 0:
            print('{}/{} -'.format(idx,data_len), batch[-1])
