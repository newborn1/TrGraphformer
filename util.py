import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splev,splrep

matplotlib.rcParams['font.family'] = 'Kaiti'

def discrete_timestep(baseTimestep, timestep) -> int:
    r"""
    离散化,取第一个作为代表,discrete是插值大小

    TODO:插入要多少才合适?
    """
    discrete = 10

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
        lat:[121.94-122.37]->[0,100]
        lon:[29.78,30.01]->[0,100]

    TODO:这里原范围和映射的范围要多少合适?需要后期调参
    '''
    left_lat, right_lat = 121.94, 122.37
    left_lon, right_lon = 29.78, 30.01
    x = (lat - left_lat) * 100 / (right_lat - left_lat)
    y = (lon - left_lon) * 100 / (right_lon - left_lon)

    return (x, y)


def fomatReadFile(_path) -> pd:
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
    # 遍历所有的行
    for index, row in df.iterrows():
        df.loc[index, 'x'], df.loc[index,
                                   'y'] = lat_lon2coor(row['x'], row['y'])
        df.loc[index, 'timestep'] = discrete_timestep(baseTimestep,
                                                      row['timestep'])

    # 按照 timestep 去重,保留第一个 !!! 否者同一帧会出现同一艘船多次
    df = df.drop_duplicates(subset=['timestep'], keep='first')
    # df.rename(columns={'timestep': 'frame_id'})

    return df


def saveToFile(base ,_path, data) -> np.void:
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


def interpolate(data,discrete)->None:
    timestep = data.loc[:,('timestep')]
    for i in range(timestep.size - 1):
        if timestep.iloc[i+1]-timestep.iloc[i] > 15*60:
            return pd.DataFrame(columns=['timesteps','mmsi','x','y'])
    mmsi = data.loc[0,('mmsi')]
    x = data.loc[:,('x')]
    y = data.loc[:,('y')]
    ipo_x = splrep(timestep,x,k = 3) # 确定x和timestep的插值关系
    ipo_y = splrep(timestep,y,k = 3)
    # 确定插值区间——取最小和最大的timestep网格化、时间对齐、去重、除去较短的轨迹
    min_timestep = int(min(timestep))
    max_timestep = int(max(timestep)) + discrete
    # 确定插值区间内需要插值的点
    timestep_range = range(*{'start':min_timestep,'stop':max_timestep,'step':discrete}.values()) # 总范围
    timesteps = [_timestep for _timestep in timestep_range if not pd.Series(_timestep).isin(timestep).values] # 范围内缺失的值
    # 确定插值节点和插值函数值
    x = splev(timesteps,ipo_x)
    y = splev(timesteps,ipo_y)
    interpolation = pd.DataFrame({'timestep':timesteps,'x':x,'y':y,'mmsi':[mmsi]*len(timesteps)})
    data = pd.concat([data,interpolation],axis=0).sort_values(by='timestep',ascending=True) # 合并然后按时间顺序排序

    return data


def visulization(data):
    r"""可视化data的三位数据并保存图片
    data:timestep、mmsi、x、y
    """
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(data.loc[:,('timestep')], data.loc[:,('x')], data.loc[:,('y')], color=['blue'])
    ax.plot3D(data.loc[:,('timestep')], data.loc[:,('x')],data.loc[:,('y')], 'blue')

    ax.set_title('{}号船舶轨迹图和轨迹点'.format(data.loc[0,('mmsi')]))
    ax.set_xlabel('时间')
    ax.set_ylabel('x坐标')
    ax.set_zlabel('y坐标')

    plt.savefig('./image/{}'.format(data.loc[0,('mmsi')]))
    # plt.show()