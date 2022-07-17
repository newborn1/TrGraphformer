# 准备数据,参考social-stgcnn和trAISformer
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """customized pytorch dataset"""

    def __init__(self, data, max_seqlen=96) -> None:
        super(TrajectoryDataset).__init__()
        r"""
        Args:
            data::=
            max_seqlen:transformer一个seq的最大长度
                    
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        self.data = data
        self.max_seqlen = max_seqlen

    def __getitem__(self, index):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
