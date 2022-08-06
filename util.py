# 存放各种工具函数
import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# from scipy.interpolate import splev, splrep

matplotlib.rcParams['font.family'] = 'Kaiti'


def discrete_timestep(baseTimestep, timestep, config) -> int:
    r"""
    离散化,取第一个作为代表,discrete是插值大小

    TODO:插入要多少才合适?
    """
    discrete = config.discrete

    baseTimestep = int(baseTimestep)
    timestep = int(timestep)

    return baseTimestep + ((timestep - baseTimestep) // discrete) * discrete


def lat_lon2coor(lat, lon):
    r'''
    输入：
    lat:经度
    lon:纬度
    返回：
    (x,y):点的坐标.lat->x,lon->y

    方法:确定经纬度范围,然后映射到0-100.
        lat:[121.94-122.37]->[0,100],一个格子大概0.47km
        lon:[29.78,30.01]->[0,100],一个格子大概0.25km

    TODO:这里原范围和映射的范围要多少合适?需要后期调参
    '''
    left_lat, right_lat = 121.94, 122.37
    left_lon, right_lon = 29.78, 30.01
    x = (lat - left_lat) * 100 / (right_lat - left_lat)
    y = (lon - left_lon) * 100 / (right_lon - left_lon)

    return (x, y)


def fomatReadFile(_path, config) -> pd:
    r"""
    输入:
    _path:要处理的文件的路径
    返回:
    df:具有论文格式的数据

    方法:
    将_path路径的读取,然后提取要的列,再对列进行分别处理
    timestep:使用discrete_timestep()函数离散化时间戳,并取前面的代表这段时间
    mmsi:不处理
    x,y:使用lat_lon2coor()函数映射到一定范围
    """
    df = pd.read_csv(_path,
                     header=None,
                     names=[
                         'mmsi', 'x', 'y', 'cog', 'true_heading', 'sog', 'rot',
                         'BaseTime', 'timestep'
                     ])

    df = df.loc[:, ('timestep', 'mmsi', 'x', 'y')]
    _path = _path.replace('\\', '/')
    baseBaseTime = _path.split('/')[2] + ' 00:00:00'
    baseTimestep = time.mktime(time.strptime(baseBaseTime,
                                             "%Y-%m-%d %H:%M:%S"))
    # 遍历所有的行，同时删去异常点(间隔1s但是和前一个相差大于0.005的点就去除)
    for index, row in df.iterrows():
        df.loc[index, 'x'], df.loc[index,
                                   'y'] = lat_lon2coor(row['x'], row['y'])
        df.loc[index, 'timestep'] = discrete_timestep(baseTimestep,
                                                      row['timestep'], config)
        dt = df.loc[index, 'timestep'] - df.loc[max(index - 1, 0), 'timestep']
        # 这里的0.4和0.2是根据一个格子的距离计算的，以1km为标准
        if (abs(df.loc[index, 'x'] - df.loc[max(index - 1, 0), 'x']) >
                config.abnormal_dx * dt or
                abs(df.loc[index, 'y'] -
                    df.loc[max(index - 1, 0), 'y'] > config.abnormal_dy * dt)):
            if dt == 0: continue
            # 把timestep变成和上一个相同的，这样执行drop的时候就会被去掉,又不会影响迭代顺序
            df.loc[index, 'timestep'] = df.loc[index - 1, 'timestep']
            # 更新为非异常的值
            df.loc[index, 'x'] = df.loc[max(index - 1, 0), 'x']
            df.loc[index, 'y'] = df.loc[max(index - 1, 0), 'y']

    # 按照 timestep 去重,保留第一个 !!! 否者同一帧会出现同一艘船多次，同时会删除异常点
    df = df.drop_duplicates(subset=['timestep'], keep='first')
    # df.rename(columns={'timestep': 'frame_id'})

    return df


def saveToFile(base, _path, data) -> np.void:
    r"""
    输入:
    base:要保存的文件的根目录。如:formatData
    _path:要保存到的路径,是csv格式的。
    data:要保存的数据。
    """
    # 替换路径前缀,建立文件夹等
    _path = _path.replace('lssddata', base).replace('\\', '/')
    dirs = '/'.join(_path.split('/')[:-1])
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    if not os.path.isfile(_path):
        data.to_csv(_path)

    return


def interpolate(data, config) -> pd.DataFrame:
    if data.shape[0] < config.interpolate_min_point:
        return pd.DataFrame(columns=['timestep', 'mmsi', 'x',
                                     'y'])  # timestep没有s
    timestep = data.loc[:, ('timestep')]
    for i in range(timestep.size - 1):
        if timestep.iloc[
                i + 1] - timestep.iloc[i] > config.interpolate_min_gap * 60:
            # if timestep.iloc[i + 1] - timestep.iloc[i] < 60*60:
            # return pd.DataFrame(columns=['timesteps','mmsi','x','y']) # 超过30分钟但是小于一小时就去除轨迹
            return pd.concat([
                interpolate(data.iloc[:i + 1], config),
                interpolate(data.iloc[i + 1:], config)
            ],
                             axis=0)  #超过30min的分成两段,但是时间不连续!!!小于30min的直接插值
    mmsi = data.iloc[0].at['mmsi']  # at用于获取单个数值,iloc用于获取一个列
    x = data.loc[:, ('x')]
    y = data.loc[:, ('y')]
    ipo_x = splrep(timestep, x, k=3)  # 确定x和timestep的插值关系
    ipo_y = splrep(timestep, y, k=3)
    # 确定插值区间——取最小和最大的timestep网格化、时间对齐、去重、除去较短的轨迹
    min_timestep = int(min(timestep))
    max_timestep = int(max(timestep)) + config.discrete
    # 确定插值区间内需要插值的点,因为是变量所以需要用*{}语法
    timestep_range = range(*{
        'start': min_timestep,
        'stop': max_timestep,
        'step': config.discrete
    }.values())  # 总范围
    timesteps = [
        _timestep for _timestep in timestep_range
        if not pd.Series(_timestep).isin(timestep).values
    ]  # 范围内缺失的值
    # 确定插值节点和插值函数值
    x = splev(timesteps, ipo_x)
    y = splev(timesteps, ipo_y)
    interpolation = pd.DataFrame({
        'timestep': timesteps,
        'x': x,
        'y': y,
        'mmsi': [mmsi] * len(timesteps)
    })
    data = pd.concat([data, interpolation],
                     axis=0).sort_values(by='timestep',
                                         ascending=True)  # 合并然后按时间顺序排序

    return data


def visulize(basedir='./interpolation', show=True):  # 所需配置不足一个就不用config参数(不直观)
    r"""可视化根目录下basedir中的数据并保存图片
    basedir下的data:timestep、mmsi、x、y
    """
    for folder in os.listdir(basedir):
        folder = os.path.join(basedir, folder)

        for file in os.listdir(folder):
            file = os.path.join(folder, file)
            # 读取数据
            data = pd.read_csv(file).loc[:, ('timestep', 'mmsi', 'x', 'y')]
            # 画图
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection='3d')
            ax.scatter3D(data.loc[:, ('timestep')],
                         data.loc[:, ('x')],
                         data.loc[:, ('y')],
                         color=['blue'])
            ax.plot3D(data.loc[:, ('timestep')], data.loc[:, ('x')],
                      data.loc[:, ('y')], 'blue')

            ax.set_title('{}号船舶轨迹图和轨迹点'.format(int(data.loc[0, ('mmsi')])))
            ax.set_xlabel('时间')
            ax.set_ylabel('x坐标')
            ax.set_zlabel('y坐标')

            plt.savefig('./image/{}'.format(int(data.loc[0, ('mmsi')])))
            if show: plt.show()


def read_epoch(path):
    data = pd.DataFrame(columns=['timestep', 'mmsi', 'x', 'y'])
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        data = pd.concat([data, pd.read_csv(filename)], axis=0)

    return data