from util import get_noise
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    r"""使用了多头注意力机制.我觉得这里不需要mask因为已经对齐了
    https://www.cnblogs.com/xiximayou/p/13978859.html
    TODO n_head是需要调整的参数
    Args:
        x:a Tensor of size (batchsize,seq*feat,nodelen)
    Return:
        y:a Tensor of size as x
    """

    def __init__(self, config) -> None:
        super(CausalSelfAttention, self).__init__()
        self.n_embd = 5
        self.n_head = 1
        self.attn_pdrop = 0.5
        self.resid_pdrop = 0.6
        assert self.n_embd % self.n_head == 0
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)
        # 正则化
        self.attn_drop = nn.Dropout(self.attn_pdrop)
        self.resid_drop = nn.Dropout(self.resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        # 下面是causal mask
        # tril函数主要用于返回一个矩阵主对角线以下的下三角矩阵，其它元素全部为0 00。当输入是一个多维张量时，返回的是同等维度的张量并且最后两个维度的下三角矩阵的。
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.max_seqlen,
                           config.max_seqlen).view(1, 1, config.max_seqlen,
                                                   config.max_seqlen)))

    def forward(self, x):  # x.size=[Batch,Feat*SeqlLen,NodeLen]
        Batch, T, NodeLen = x.size()

        # 增加一个维度来存放复制出来的"head",但是需要调整顺序方便计算,见https://www.cnblogs.com/xiximayou/p/13978859.html
        k = self.key(x).view(Batch, T, self.n_head,
                             NodeLen // self.n_head).transpose(1, 2)
        q = self.key(x).view(Batch, T, self.n_head,
                             NodeLen // self.n_head).transpose(1, 2)
        v = self.key(x).view(Batch, T, self.n_head,
                             NodeLen // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(Batch, T, NodeLen)

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    r"""Transformer的一个block,通过他可以来构建n个模块
    TODO 后期需要设置n的参数
    Args:
        x:a Tensor of size (batchsize,seq*feat,nodelen)
    Return:
        x:a Tensor of size as input——x
    """

    def __init__(self, config) -> None:
        super(Block, self).__init__()
        self.n_embd = 5  # 输入数据的维度
        self.resid_pdrop = 0.8  # 防止过拟合
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        # 这个才是整个Block的重点,进行Attention的计算
        self.attn = CausalSelfAttention(config)
        # 处理提取优化输入数据
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(self.resid_pdrop),
        )

    def forward(self, x):
        x += self.attn(self.ln1(x))
        x += self.mlp(self.ln2(x))

        return x


class TrGraphformerEncoderLayer(nn.Module):
    r"""
    Transformer for 结果处理后变成图的的AIS的轨迹
    """

    def __init__(self, config) -> None:
        super(TrGraphformerEncoderLayer, self).__init__()

        self.n_embd = 5
        self.maxseqlen = 8
        self.embd_pdrop = 0.8
        self.n_layer = config.n_layer
        self.drop = nn.Dropout(self.embd_pdrop)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(self.n_layer)])

        # decoder head——未实现decoder
        self.ln_f = nn.LayerNorm(self.n_embd)

    def forward(self, x, masks=None, with_targets=False):
        """
        Args:
            x:a Tensor of size (batchsize,feat_len,seq_len,node_len)
            TODO 不知道需不需要正则化
            masks: a Tensor of the same size of x.0是padding
        Return:
            y:a Tensor of size as x 
        """

        # 打平维度变成二维信息
        batchsize, feat_len, seq_len, node_len = x.size()
        seqlen = feat_len * seq_len
        x = x.view(batchsize, seqlen, node_len)
        assert seqlen <= self.maxseqlen, "Cannot forward,model block size is exhausted."

        # 先合起来，后面再拆开dim=-1是什么意思，要不要加embedding层
        token_embeddings = x

        # TODO 这里需要修改
        position_embeddings = self.pos_emb[:, :seqlen, :]

        fea = self.drop(token_embeddings + position_embeddings)
        fea = self.blocks(fea)
        logits = self.ln_f(fea)
        # 这里需要还原数据

        return logits


class TrGraphformerEncoder(nn.Module):
    r"""由 n 个encoderlayer组成
    Args:
        encoder_layer:要复制的层
        num_layers:要复制的层数
        norm:正则化层(可选参数)
    """

    def __init__(self, encoder_layer, num_layers, norm=None) -> None:
        super(TrGraphformerEncoder, self).__init__()
        self.layers = nn.Sequential(
            *[encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""
        Args:
            src: 需要encode的序列
            mask: src序列的mask
            src_key_padding_mask: mask掉每一个 batch 的值
        
        """
        output = self.layers(src)

        if self.norm:
            output = self.norm(output)

        return output


class TrGraphformerModel(nn.Module):

    def __init__(self, config) -> None:
        super(TrGraphformerModel, self).__init__()
        # self.src_mask = config.src_mask
        self.encoder_layers = TrGraphformerEncoderLayer(config)
        # self.transformer_encoder = TrGraphformerEncoder(config)

    def forward(self, inputs, iftest=False):
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_ship_num = inputs
        num_ship = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_ship, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_ship, 32).cuda()

        return inputs[0]


class STAR(torch.nn.Module):

    def __init__(self, config) -> None:
        super(STAR, self).__init__()

        # 设置参数
        self.embedding_size = config.embedding_size
        self.output_size = config.output_size
        self.dropout_prob = config.dropout_prob
        self.config = config

        self.temporal_encoder_layer = TrGraphformerEncoderLayer(d_model=32,
                                                                nhead=8)

        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward
        nlayers = 2  # the number of nn.TrGraphformerEncoderLayer in nn.TrGraphformerEncode
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value

        # Linear layer:将位置坐标映射到embedding中(分为时间和空间两个部分)
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)

        self.spatial_encoder1 = TrGraphformerModel(emsize, nhead, nhid,
                                                   nlayers, dropout)
        self.spatial_encoder2 = TrGraphformerModel(emsize, nhead, nhid,
                                                   nlayers, dropout)

        self.temporal_encoder1 = TrGraphformerEncoder(
            self.temporal_encoder_layer, 1)
        self.temporal_encoder2 = TrGraphformerEncoder(
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
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]

            # encoder1/Temporal Transformer
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]
            temporal_input_embedded = temporal_input_embedded[:-1]

            # 融合时间和空间两个特征,concat和FC部分
            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)


            #=================================================================================================#

            # encoder2/Spatial Transformer
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)

            # encoder2/Temporal Transformer将encoder1的学到的时间特性和这一层学到的空间特性进行学习(TODO 为什么是上一层的时间,论文有没有提到)
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]

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
