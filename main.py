# 这个模块用于生成预处理后的数据并保存在文件夹中,同时可视化数据
import os
from util import *

if __name__ == '__main__':
    dataDir = './lssddata/'
    all_folders = os.listdir(dataDir)
    all_folders = [os.path.join(dataDir, _path) for _path in all_folders]

    xMaxs, xMins, yMaxs, yMins = [], [], [], []
    times = 1

    for folder in all_folders:
        all_files = os.listdir(folder)
        all_files = [os.path.join(folder, _path) for _path in all_files]

        for file in all_files:
            # 预处理——网格化、时间对齐、去重、出去较短的轨迹
            data = fomatReadFile(file)
            # print(data)
            saveToFile('formatData', file, data)
            # 插值处理
            if data.shape[0] < 30: continue  # 小于30*10s的路线删除 TODO 这个设置多少好
            data = interpolate(data, 10)
            if data.empty is False:
                saveToFile('interpolation', file, data)
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
    visulize('./interpolation', True)
