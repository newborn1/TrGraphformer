# 这个模块用于生成预处理后的数据并保存在文件夹中,同时可视化数据
import os
from util import *
from config import Config as config

if __name__ == '__main__':
    dataDir = './lssddata/'
    xMaxs, xMins, yMaxs, yMins = [], [], [], []
    times = 1

    for folder in os.listdir(dataDir):
        folder = os.path.join(dataDir, folder)

        for file in os.listdir(folder):
            file = os.path.join(folder, file)
            # 预处理——网格化、时间对齐、去重、出去较短的轨迹
            data = fomatReadFile(file, config)
            # print(data)
            saveToFile('formatData', file, data)
            # 插值处理
            data = interpolate(data, config)
            if data.empty is False:
                saveToFile('data/eth', file, data)
            """
            # 这是测试找范围的
            x = data.loc[:, ('x')]
            y = data.loc[:, ('y')]
            yMaxs.append(max(y))
            yMins.append(min(y))
            xMaxs.append(max(x))
            xMins.append(min(x))

            if (len(yMaxs) == 1000 * times):
                print('处理', times * 1000, '个文件')
                print('x的范围', max(xMaxs), min(xMins))
                print('y的范围', max(yMaxs), min(yMins), end='\n\n')
                times += 1
            """

    # 可视化部分
    visulize('./data/eth', config.pre_show)
