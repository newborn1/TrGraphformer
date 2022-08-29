from util import get_noise
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

# nn.MultiheadAttention() 替换他提到的东西
# nn.TransformerEncoder()
# nn.TransformerEncoderLayer()
# nn.Transformer()

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src,
                                    src,
                                    src,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(
                self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)]) # 深复制
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = [] # 好像没有什么用

        for i in range(self.num_layers):
            output, attn = self.layers[i](
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output

class TransformerModel(nn.Module):
    r"""
    不包含decoder模块
    """

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
        n_mask = n_mask.float().masked_fill(n_mask == 0.,
                                            float(-1e20)).masked_fill(
                                                n_mask == 1., float(0.0)) # 0用负无穷替代,1用0替代
        output = self.transformer_encoder(src, mask=n_mask)

        return output


class STAR(torch.nn.Module):

    def __init__(self, config) -> None:
        super(STAR, self).__init__()

        # 设置参数
        self.embedding_size = config.embedding_size
        self.output_size = 2
        self.dropout_prob = config.dropout_prob
        self.config = config

        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32,
                                                                nhead=8)

        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward
        nlayers = 2  # the number of nn.TrGraphformerEncoderLayer in nn.TrGraphformerEncode
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        # Linear layer:将位置坐标映射到embedding中(分为时间和空间两个部分)
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)

        self.spatial_encoder1 = TransformerModel(emsize, nhead, nhid,
                                                   nlayers, dropout)
        self.spatial_encoder2 = TransformerModel(emsize, nhead, nhid,
                                                   nlayers, dropout)

        self.temporal_encoder1 = TransformerEncoder(
            self.temporal_encoder_layer, 1)
        self.temporal_encoder2 = TransformerEncoder(
            self.temporal_encoder_layer, 1)

        # Linear layer to output and fusion
        self.output_layer = nn.Linear(48, 2)
        self.fusion_layer = nn.Linear(64, 32)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_in1 = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)

    def forward(self, inputs, iftest=False):
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        # 和nodes_norm的shape一样
        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        # Graph Memory,用于保存数据(保存32/2=16组数据?)
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

        noise = get_noise((1, 16), 'gaussian')  # 加入高斯噪声
        for frame_id in range(self.config.max_seqlen - 1):
            if frame_id >= self.config.obs_len and iftest:
                pass
            else:
                # 获取有效的节点信息,加快训练速度,同时
                node_index = self.get_node_index(seq_list[:frame_id + 1])
                # 获取第frame_id帧有效船只的所有邻居信息
                nei_list = nei_lists[frame_id, node_index, :]
                # 获取有效船的有效邻居节点
                nei_list = nei_list[:, node_index]
                # 获得有效节点的信息
                nodes_current = nodes_norm[:frame_id + 1, node_index]
                # 在同一个场景下使用平均值来正则化绝对值坐标
                st_ed = [(0, nodes_current.shape[1])]
                node_abs = self.mean_normalize_abs(
                    nodes_abs[:frame_id + 1, node_index], st_ed)

            # Input Embedding
            # TODO 能不能把他放在一起搭建小网络
            temporal_input_embedded = self.dropout_in1(
                self.relu(self.input_embedding_layer_temporal(nodes_current)))
            if frame_id != 0: # 从内存池中读取信息
                temporal_input_embedded = temporal_input_embedded.clone()
                temporal_input_embedded[:frame_id] = GM[:frame_id, node_index]

            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs))) # 这里用的是标准化后的,和前面的不太一样

            # encoder1/Spatial Transformer
            spatial_input_embedded = self.spatial_encoder1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]

            # encoder1/Temporal Transformer
            temporal_input_embedded_last = self.temporal_encoder1(temporal_input_embedded)[-1]
            temporal_input_embedded = temporal_input_embedded[:-1]

            # 融合时间和空间两个特征,concat和FC部分
            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)


            #=================================================================================================#

            # encoder2/Spatial Transformer
            spatial_input_embedded = self.spatial_encoder2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)

            # encoder2/Temporal Transformer将encoder1的学到的时间特性和这一层学到的空间特性进行学习(TODO 为什么是上一层的时间,论文有没有提到)
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = self.temporal_encoder2(temporal_input_embedded)[-1]

            # TODO 为什么要加入noise
            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)

            # decoder FC
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            outputs[frame_id, node_index] = outputs_current
            GM[frame_id, node_index] = temporal_input_embedded

        return outputs

    def get_node_index(self, seq_list):
        r'''返回值用于索引确定船的信息
        Args:
            seq_list:shape=[F(总帧数),N(每一帧下的总人数)],seq_list[f,n]:f帧下第n个人是否存在
        Return:
            node_indices:返回所有的从开始到当前帧都一直存在的所有船的bool索引
            shape=[N(每一帧下的行人总数)],node_indices[n]第n艘船是否从开头到现在一直存在
        '''
        for idx, frame_id in enumerate(seq_list):
            if idx == 0:
                node_indices = frame_id > 0
            else:
                node_indices *= (frame_id > 0)

        return node_indices

    def mean_normalize_abs(self, node_abs, st_ed):
        r'''
        Args:
            node_abs:
            st_ed:每个场景的开始和结束下标
        Return:
            node_abs:正则化后的船坐标
        '''
        node_abs = node_abs.permute(1, 0, 2)  # 转换维度方便处理坐标(最后一维方便计算)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)
