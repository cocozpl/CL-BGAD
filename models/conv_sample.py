
from typing import List, Optional, Tuple
import torch

from torch import Tensor
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_geometric.typing import PairTensor, OptTensor

from models.sampler import BEANAdjacency

# 图中的采样操作
class BEANConvSample(torch.nn.Module):
    def __init__(
        self,
        # Optional可以是一个整数也可以是None（表示该元素缺失）
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: Tuple[int, int, Optional[int]],
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.input_has_edge_channel = len(in_channels) == 3
        self.output_has_edge_channel = len(out_channels) == 3
        # 定义节点卷积层：v2u_conv和u2v_conv
        # 加注意力机制要修改成BEANConvNodeWithAttention
        self.v2u_conv = BEANConvNode(
            in_channels,
            # out_channels[0] 表示的是u节点的输出通道数
            out_channels[0],
            flow="v->u",
            node_self_loop=node_self_loop,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )

        self.u2v_conv = BEANConvNode(
            in_channels,
            # out_channels[1] 表示的是v节点的输出通道数
            out_channels[1],
            flow="u->v",
            node_self_loop=node_self_loop,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )

        if self.output_has_edge_channel:
            self.e_conv = BEANConvEdge(
                in_channels,
                # out_channels[2]：边的输出特征维度
                out_channels[2],
                node_self_loop=node_self_loop,
                normalize=normalize,
                bias=bias,
                **kwargs,
            )

    def forward(
        self,
        # xu表示u节点的源特征和目标特征
        xu: PairTensor,
        xv: PairTensor,
        adj: BEANAdjacency,
        xe: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # source and target
        xus, xut = xu
        xvs, xvt = xv

        # xe
        if xe is not None:
            xe_e, xe_v2u, xe_u2v = xe
        # 第一个 adj 是 BEANAdjacency 对象，表示整个图结构中的邻接信息。
        # 第二个 adj 是 adj_v2u 属性的邻接矩阵，用于描述 v 到 u 节点之间的连接关系
        out_u = self.v2u_conv((xut, xvs), adj.adj_v2u.adj, xe_v2u)
        out_v = self.u2v_conv((xus, xvt), adj.adj_u2v.adj, xe_u2v)

        out_e = None
        if self.output_has_edge_channel:
            out_e = self.e_conv((xut, xvt), adj.adj_e.adj, xe_e)

        return out_u, out_v, out_e

# 加注意力机制，在BEANConvNode中实现一个新的消息传递机制，使用注意力权重来对邻居节点的信息进行加权聚合。
class BEANConvNode(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: int,
        flow: str = "v->u",
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        agg: List[str] = ["mean", "max"],
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flow = flow
        self.node_self_loop = node_self_loop
        self.normalize = normalize
        self.agg = agg

        self.input_has_edge_channel = len(in_channels) == 3

        n_agg = len(agg)
        # calculate in channels
        if self.input_has_edge_channel:
            if self.node_self_loop:
                if flow == "v->u":
                    self.in_channels_all = (
                        # in_channels[0]是u节点自身的特征，通常不需要聚合操作直接使用
                        # in_channels[1]是v节点的特征维度，它代表的是从v节点传递到u节点的消息。在消息传递过程中，v节点的特征会通过不同的聚合方式（比如mean、max）进行处理。
                        # in_channels[2]是边的特征维度，表示从边传递的特征。
                        # 所以如果想修改
                        in_channels[0] + n_agg * in_channels[1] + n_agg * in_channels[2]
                    )
                else:
                    self.in_channels_all = (
                        n_agg * in_channels[0] + in_channels[1] + n_agg * in_channels[2]
                    )
            else:
                if flow == "v->u":
                    self.in_channels_all = (
                        n_agg * in_channels[1] + n_agg * in_channels[2]
                    )
                else:
                    self.in_channels_all = (
                        n_agg * in_channels[0] + n_agg * in_channels[2]
                    )
        else:
            if self.node_self_loop:
                if flow == "v->u":
                    self.in_channels_all = in_channels[0] + n_agg * in_channels[1]
                else:
                    self.in_channels_all = n_agg * in_channels[0] + in_channels[1]
            else:
                if flow == "v->u":
                    self.in_channels_all = n_agg * in_channels[1]
                else:
                    self.in_channels_all = n_agg * in_channels[0]
        # 线性层和归一化，self.lin表示线性层，将输入映射到指定的out_channels维度
        self.lin = Linear(self.in_channels_all, out_channels, bias=bias)
        self.attn_weight = nn.Parameter(torch.Tensor(in_channels[1], 1))

        nn.init.xavier_uniform_(self.attn_weight)
        # 在特征维度上对输出进行归一化
        if normalize:
            self.bn = nn.BatchNorm1d(out_channels)
        # 初始化线性层的权重和偏置参数
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.xavier_uniform_(self.attn_weight)  # Reset attention weights

    def forward(self, x: PairTensor, adj: SparseTensor, xe: OptTensor = None) -> Tensor:
        """"""
        # 检查输入是否包含边特征，并确保xe和输入设置相匹配
        assert self.input_has_edge_channel == (xe is not None)

        # propagate_type: (x: PairTensor)
        # 调用MessagePassing类的propagate方法进行消息传递
        out = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out = self.lin(out)
        if self.normalize:
            # 归一化操作
            out = self.bn(out)

        return out

    # 消息传递的核心方法，用于计算从源节点到目标节点的消息和聚合操作
    # 如果我要在encoder添加注意力机制需要在这里修改集合的机制，比如要修改矩阵乘法聚合和 scatter 聚合变成注意力机制的
    def message_and_aggregate(
        self, adj: SparseTensor, x: PairTensor, xe: OptTensor
    ) -> Tensor:

        xu, xv = x
        adj = adj.set_value(None, layout=None)

        # Attention mechanism
        attention_scores = torch.matmul(xv, self.attn_weight).squeeze(-1)  # 计算邻居节点的注意力得分
        attention_weights = torch.softmax(attention_scores, dim=0)  # 归一化得到注意力权重

        # 调整维度，确保 attention_weights 与 xv 形状一致
        attention_weights = attention_weights.unsqueeze(-1).expand_as(xv)

        # 对邻居节点的消息进行加权
        if self.flow == "v->u":
            msg_v2u_list = [matmul(adj, attention_weights * xv, reduce=ag) for ag in self.agg]

            # messages edge to node
            if xe is not None:
                msg_e2u_list = [
                    scatter(xe, adj.storage.row(), dim=0, reduce=ag) for ag in self.agg
                ]

            # collect all msg
            if xe is not None:
                if self.node_self_loop:
                    msg_2u = torch.cat((xu, *msg_v2u_list, *msg_e2u_list), dim=1)
                else:
                    msg_2u = torch.cat((*msg_v2u_list, *msg_e2u_list), dim=1)
            else:
                if self.node_self_loop:
                    msg_2u = torch.cat((xu, *msg_v2u_list), dim=1)
                else:
                    msg_2u = torch.cat((*msg_v2u_list,), dim=1)

            return msg_2u

        else:
            # Node U to node V
            attention_scores = torch.matmul(xu, self.attn_weight).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=0)
            # 调整维度，确保 attention_weights 与 xu 形状一致
            attention_weights = attention_weights.unsqueeze(-1).expand_as(xu)


            msg_u2v_list = [matmul(adj.t(), attention_weights * xu, reduce=ag) for ag in self.agg]

            if xe is not None:
                msg_e2v_list = [
                    scatter(xe, adj.storage.col(), dim=0, reduce=ag) for ag in self.agg
                ]

            if xe is not None:
                if self.node_self_loop:
                    msg_2v = torch.cat((xv, *msg_u2v_list, *msg_e2v_list), dim=1)
                else:
                    msg_2v = torch.cat((*msg_u2v_list, *msg_e2v_list), dim=1)
            else:
                if self.node_self_loop:
                    msg_2v = torch.cat((xv, *msg_u2v_list), dim=1)
                else:
                    msg_2v = torch.cat((*msg_u2v_list,), dim=1)

            return msg_2v


class BEANConvEdge(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: int,
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.input_has_edge_channel = len(in_channels) == 3

        if self.input_has_edge_channel:
            self.in_channels_e = in_channels[0] + in_channels[1] + in_channels[2]
        else:
            self.in_channels_e = in_channels[0] + in_channels[1]

        self.lin_e = Linear(self.in_channels_e, out_channels, bias=bias)

        if normalize:
            self.bn_e = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_e.reset_parameters()

    def forward(self, x: PairTensor, adj: SparseTensor, xe: Tensor) -> Tensor:
        """"""

        # propagate_type: (x: PairTensor)
        out_e = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out_e = self.lin_e(out_e)

        if self.normalize:
            out_e = self.bn_e(out_e)

        return out_e

    def message_and_aggregate(
        self, adj: SparseTensor, x: PairTensor, xe: OptTensor
    ) -> Tensor:

        xu, xv = x
        adj = adj.set_value(None, layout=None)

        # collect all msg (including self loop)
        if xe is not None:
            msg_2e = torch.cat(
                (xe, xu[adj.storage.row()], xv[adj.storage.col()]), dim=1
            )
        else:
            msg_2e = torch.cat((xu[adj.storage.row()], xv[adj.storage.col()]), dim=1)

        return msg_2e
