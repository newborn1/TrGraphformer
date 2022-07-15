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
        super().__init__()
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

    def __init__(self) -> None:
        super().__init__()
        self.n_embd = 5  # 输入数据的维度
        self.resid_pdrop = 0.8  # 防止过拟合
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        # 这个才是整个Block的重点,进行Attention的计算
        self.attn = CausalSelfAttention()
        # 处理提取优化输入数据
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.ReLu(inplace=True),
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
        self.blocks = nn.Sequential(*[Block() for _ in range(self.n_layer)])

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
        self.src_mask = config.src_mask
        encoder_layers = TrGraphformerEncoderLayer(config)
        self.transformer_encoder = TrGraphformerEncoder(config)


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

        outputs = torch.zeros()
