### 问题：

- 1、输入 transformer 的数据维度长度不是确定的怎么办
- 2、这个数据长度多少去除好:

  目前选的是插值后的数据小于 30min 的轨迹直接删除(discrete 为 10 的情况下)

- 3、注意静止!

  x 方向极差<0.05 且 y 方向极差<0.05 的视为静止

- 4、数据异常点如何去除

  小于相邻十秒小于 0.01 的如何删除即可

- 5、插值间隔太大不准确问题:

  大于 30min 的拆成两段,小于 30min 的直接进行插值

- 6、在 social-stgcnn 中在 seq_len 个帧中很少有都出现 seq_len 帧的船,导致很多船都被抛弃了,基本是 15:1

### 注:

- 1、这里的输入结构综合考虑了 social-stgcnn 和 STAR 两个,以 STAR 为主为文件的结构，输入代码参考为 social-stgcnn
