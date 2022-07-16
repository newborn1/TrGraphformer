from util import *
import numpy as np
import pandas as pd

basedir = './formatData'
folders = [os.path.join(basedir, _path) for _path in os.listdir(basedir)]
for folder in folders:
    all_files = os.listdir(folder)
    all_files = [os.path.join(folder, _path) for _path in all_files]

    for file in all_files:
        data = pd.read_csv(file)
        data = data.loc[:, ('timestep', 'mmsi', 'x', 'y')]
        visulization(data)
