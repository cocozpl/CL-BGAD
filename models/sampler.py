
import torch
from torch import Tensor

import math

from torch_sparse import SparseTensor
from torch_sparse.storage import SparseStorage

from typing import List, NamedTuple, Optional, Tuple, Union

from utils.sparse_combine import spadd
from utils.sprand import sprand


class EdgePredictionSampler:
    def __init__(
        self,
        adj: SparseTensor,
        n_random: Optional[int] = None, # 随机采样的边的数量，根据 mult 和图中正样本边的数量来确定
        mult: Optional[float] = 2.0,  # 用于确定 n_random 的系数
    ):
        self.adj = adj

        if n_random is None:
            n_pos = adj.nnz() # 非零元素的数量，代表图中正样本边的数量（即实际存在的边数）
            n_random = mult * n_pos

        self.adj = adj
        self.n_random = n_random

    def sample(self):
        rnd_samples = sprand(self.adj.sparse_sizes(), self.n_random)
        rnd_samples.fill_value_(-1) # 边的值设为 -1,表示这些是负样本边（不存在的边）
        rnd_samples = rnd_samples.to(self.adj.device())

        pos_samples = self.adj.fill_value(2) # 设为 2，表示这些是正样本边（存在的边）

        samples = spadd(rnd_samples, pos_samples) #将 rnd_samples 和 pos_samples 合并
        samples.set_value_(
            torch.minimum(
                samples.storage.value(), torch.ones_like(samples.storage.value())
            ),
            layout="coo",
        )

        return samples


### REGION Neighbor Sampling
# 给定 u 节点，采样其对应的 v 邻居节点
def sample_v_given_u(
    adj: SparseTensor,
    u_indices: Tensor,# 需要采样的 u 节点的索引
    prev_v: Tensor,# 前一轮中采样得到的 v 节点索引，用于当前采样
    num_neighbors: int,
    replace=False,
) -> Tuple[SparseTensor, Tensor]:

    # to homogenous adjacency
    nu, nv = adj.sparse_sizes()
    # 使得u和v节点被合并到同一个矩阵中，方便统一处理采样
    adj_h = SparseTensor(
        row=adj.storage.row(),# 行索引
        col=adj.storage.col() + nu, # 列索引
        value=adj.storage.value(),
        sparse_sizes=(nu + nv, nu + nv),
    )
    # 从合并后的邻接矩阵中,基于 u_indices 和 prev_v（偏移后的 v 节点索引）进行邻居采样
    # res_adj_h是采样后的邻接矩阵（同质矩阵） res_id是采样得到的节点的索引
    res_adj_h, res_id = adj_h.sample_adj(
        torch.cat([u_indices, prev_v + nu]),
        num_neighbors=num_neighbors,
        replace=replace,
    )

    ni = len(u_indices) # 计算采样的 u 节点的数量。表示将处理多少个 u 节点。
    v_indices = res_id[ni:] - nu #从采样得到的节点索引中提取出 v 节点的索引，并减去偏移量 nu
    res_adj = res_adj_h[:ni, ni:] # 表示从采样的 u 节点到采样的 v 节点之间的连接关系。

    return res_adj, v_indices


def sample_u_given_v(
    adj: SparseTensor,
    v_indices: Tensor,
    prev_u: Tensor,
    num_neighbors: int,
    replace=False,
) -> Tuple[SparseTensor, Tensor]:

    # to homogenous adjacency
    res_adj_t, u_indices = sample_v_given_u(
        adj.t(), v_indices, prev_u, num_neighbors=num_neighbors, replace=replace
    )

    return res_adj_t.t(), u_indices

# 编码有向图的邻接关系时，方便传递和处理数据
class DirectedAdj(NamedTuple):
    adj: SparseTensor
    u_id: Tensor
    v_id: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]
    flow: str

    def to(self, *args, **kwargs):
        adj = self.adj.to(*args, **kwargs)
        u_id = self.u_id.to(*args, **kwargs)
        v_id = self.v_id.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return DirectedAdj(adj, u_id, v_id, e_id, self.size, self.flow)


class BEANAdjacency(NamedTuple):
    adj_v2u: DirectedAdj
    adj_u2v: DirectedAdj
    adj_e: Optional[DirectedAdj]

    def to(self, *args, **kwargs):
        adj_v2u = self.adj_v2u.to(*args, **kwargs)
        adj_u2v = self.adj_u2v.to(*args, **kwargs)
        adj_e = None
        if self.adj_e is not None:
            adj_e = self.adj_e.to(*args, **kwargs)
        return BEANAdjacency(adj_v2u, adj_u2v, adj_e)


class BipartiteNeighborSampler(torch.utils.data.DataLoader):
    def __init__(
        self,
        adj: SparseTensor,
        n_layers: int,
        num_neighbors_u: Union[int, List[int]],
        num_neighbors_v: Union[int, List[int]],
        base: str = "u", # 从u开始采样
        n_other_node: int = -1, #表示从另一类节点（v 或 u）中采样的数量，如果n_other_node == -1，根据u或v节点之间的比例自动推断出需要采样的另一类节点数量
        neg_samples: int = 0,  # 新增参数
        **kwargs
    ):

        adj = adj.to("cpu")

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.adj = adj
        self.n_layers = n_layers
        self.base = base
        self.n_other_node = n_other_node
        self.neg_samples = neg_samples  # 存储负采样数量

        if isinstance(num_neighbors_u, int):
            num_neighbors_u = [num_neighbors_u for _ in range(n_layers)]
        if isinstance(num_neighbors_v, int):
            num_neighbors_v = [num_neighbors_v for _ in range(n_layers)]
        self.num_neighbors_u = num_neighbors_u
        self.num_neighbors_v = num_neighbors_v

        if base == "u":  # start from u
            item_idx = torch.arange(adj.sparse_size(0))
        elif base == "v":  # start from v instead
            item_idx = torch.arange(adj.sparse_size(1))
        elif base == "e":  # start from e instead
            item_idx = torch.arange(adj.nnz())
        else:  # start from u default
            item_idx = torch.arange(adj.sparse_size(0))

        value = torch.arange(adj.nnz())
        adj = adj.set_value(value, layout="coo")
        self.__val__ = adj.storage.value()

        # transpose of adjacency
        self.adj = adj
        self.adj_t = adj.t()

        # homogenous graph adjacency matrix
        self.nu, self.nv = self.adj.sparse_sizes()
        self.adj_homogen = SparseTensor(
            row=self.adj.storage.row(),
            col=self.adj.storage.col() + self.nu,
            value=self.adj.storage.value(),
            sparse_sizes=(self.nu + self.nv, self.nu + self.nv),
        )
        self.adj_t_homogen = SparseTensor(
            row=self.adj_t.storage.row(),
            col=self.adj_t.storage.col() + self.nv,
            value=self.adj_t.storage.value(),
            sparse_sizes=(self.nu + self.nv, self.nu + self.nv),
        )

        super(BipartiteNeighborSampler, self).__init__(
            # collate_fn 函数负责将这些数据样本（例如图数据中的节点或边）打包在一起，形成最终的批次
            item_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs
        )

    def sample_v_given_u(
        self, u_indices: Tensor, prev_v: Tensor, num_neighbors: int
    ) -> Tuple[SparseTensor, Tensor]:

        res_adj_h, res_id = self.adj_homogen.sample_adj(
            torch.cat([u_indices, prev_v + self.nu]),
            num_neighbors=num_neighbors,
            replace=False,
        )

        ni = len(u_indices)
        v_indices = res_id[ni:] - self.nu
        res_adj = res_adj_h[:ni, ni:]

        return res_adj, v_indices

    def sample_u_given_v(
        self, v_indices: Tensor, prev_u: Tensor, num_neighbors: int
    ) -> Tuple[SparseTensor, Tensor]:

        # start = time.time()
        res_adj_h, res_id = self.adj_t_homogen.sample_adj(
            torch.cat([v_indices, prev_u + self.nv]),
            num_neighbors=num_neighbors,
            replace=False,
        )
        # print(f"adjoint sampling : {time.time() - start} s")

        ni = len(v_indices)
        u_indices = res_id[ni:] - self.nv
        res_adj = res_adj_h[:ni, ni:]

        return res_adj.t(), u_indices

    def adjacency_from_samples(
        self, adj: SparseTensor, u_id: Tensor, v_id: Tensor, flow: str
    ) -> DirectedAdj:

        e_id = adj.storage.value()
        size = adj.sparse_sizes()
        if self.__val__ is not None:
            adj.set_value_(self.__val__[e_id], layout="coo")
        # 提取采样得到的边的值（边的ID）和邻接矩阵的大小。
        return DirectedAdj(adj, u_id, v_id, e_id, size, flow)

    def combine_adjacency(
        self, v2u_adj: SparseTensor, u2v_adj: SparseTensor, e_adj: SparseTensor
    ) -> SparseTensor:

        # start = time.time()
        nu = u2v_adj.sparse_size(0) # 行数sparse_size(0)对应的是源节点（u节点）的数量
        nv = v2u_adj.sparse_size(1) #

        row = torch.cat(
            # dim=-1表示沿着索引向量进行连接，形成更长的索引向量。
            [e_adj.storage.row(), v2u_adj.storage.row(), u2v_adj.storage.row()], dim=-1
        )
        col = torch.cat(
            [e_adj.storage.col(), v2u_adj.storage.col(), u2v_adj.storage.col()], dim=-1
        )
        value = torch.cat(
            # dim=0：表示沿着第一个维度拼接
            [e_adj.storage.value(), v2u_adj.storage.value(), u2v_adj.storage.value()],
            dim=0,
        )
        # fl是用来表示边的类型（方向）的标记，不是常规意义的权重。这里使用了1、2、4来分别标记不同类型的边：
        fl = torch.cat(
            [
                torch.ones(e_adj.nnz()), # 1 表示 e_adj 中的边（原始边）
                2 * torch.ones(v2u_adj.nnz()), # 2 表示从 v 到 u 的边
                4 * torch.ones(u2v_adj.nnz()),  # 4 表示从 u 到 v 的边
            ]
        )

        storage = SparseStorage(
            row=row, col=col, value=value, sparse_sizes=(nu, nv), is_sorted=False
        )
        # coalesce(reduce="mean") 这个方法会将重复的边合并，并对它们的值取平均值
        # 只是想在节点或边的聚合过程中使用 LSTM： 那么 coalesce(reduce="mean") 不需要修改
        storage = storage.coalesce(reduce="mean")
        # 使用reduce = "sum" 来合并边的标记。如果一条边出现在多个方向上，标记值会进行求和操作
        fl_storage = SparseStorage(
            row=row, col=col, value=fl, sparse_sizes=(nu, nv), is_sorted=False
        )
        # 这里的reduce需要改成LSTM或者池化
        fl_storage = fl_storage.coalesce(reduce="sum") # 表示如果某条边同时存在于多个方向，将其标记值相加

        # res 合并后的邻接矩阵，包括所有边的信息。
        res = SparseTensor.from_storage(storage)
        #  表示每条边的类型（e_adj、v2u_adj 或 u2v_adj）
        flag = SparseTensor.from_storage(fl_storage)

        # print(f"combine adj : {time.time() - start} s")

        return res, flag

    def sample_negative(self, pos_edges: Tensor, num_neg: int) -> Tensor:
        # 实现负采样逻辑
        num_nodes = self.adj.sparse_sizes()[0]
        neg_edges = torch.randint(0, num_nodes, (len(pos_edges), num_neg))
        return neg_edges

    # 从图结构（Graph）中进行采样，并获取u节点和v节点之间的邻接矩阵信息
    # 接收 batch 作为输入，batch 是一个批处理节点（可能是 u 或 v 节点的索引）
    def sample(self, batch):

        # start = time.time()

        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        # 计算当前批次的大小，即批量节点的数量
        batch_size: int = len(batch)

        # calculate batch_size for another node
        # 没有显式地指定 n_other_node，即未指定另一类节点的数量，那么根据图结构的 u 和 v 节点的比例自动推断出来
        if self.n_other_node == -1 and self.base in ["u", "v"]:
            # do proportional
            nu, nv = self.adj.sparse_sizes()
            if self.base == "u":
                self.n_other_node = int(math.ceil((nv / nu) * batch_size))
            elif self.base == "v":
                self.n_other_node = int(math.ceil((nu / nv) * batch_size))

        ## get the other indices创建一个空的Tensor，用于后续存储节点索引
        empty_list = torch.tensor([], dtype=torch.long)
        if self.base == "u":
            # 如果 base 是 u 节点，开始处理 v 节点的采样
            # get the base node for v
            u_indices = batch  ## 将批次中的节点视为 u 节点的索引
            # 根据 u 节点采样与其连接的 v 节点
            # res_adj是 u 和 v 之间的邻接矩阵，res_id是采样得到的 v 节点索引
            res_adj, res_id = self.sample_v_given_u(
                u_indices, empty_list, num_neighbors=self.num_neighbors_u[0]
            )
            # 随机打乱采样的 v 节点，并从中随机选取 n_other_node 个节点
            rand_id = torch.randperm(len(res_id))[: self.n_other_node]
            v_indices = res_id[rand_id] # 获取选中的 v 节点索引。
            e_adj = res_adj[:, rand_id] # 获取u和选中v节点之间的邻接矩阵
        elif self.base == "v":
            # get the base node for u
            v_indices = batch
            res_adj, res_id = self.sample_u_given_v(
                v_indices, empty_list, num_neighbors=self.num_neighbors_v[0]
            )
            rand_id = torch.randperm(len(res_id))[: self.n_other_node]
            u_indices = res_id[rand_id]
            e_adj = res_adj[rand_id, :]
        elif self.base == "e":
            # get the base node for u and v
            row = self.adj.storage.row()[batch] #获取当前批次中边的行索引（对应 u 节点）
            col = self.adj.storage.col()[batch] #获取当前批次中边的列索引（ v 节点）
            unique_row, invidx_row = torch.unique(row, return_inverse=True) #获取唯一的 v 节点及其索引
            unique_col, invidx_col = torch.unique(col, return_inverse=True)

            # 为唯一的 u 节点生成重新索引的 ID
            reindex_row_id = torch.arange(len(unique_row))
            reindex_col_id = torch.arange(len(unique_col))
            reindex_row = reindex_row_id[invidx_row]
            reindex_col = reindex_col_id[invidx_col]

            # 根据重新索引的u和v节点构造新的邻接矩阵
            e_adj = SparseTensor(row=reindex_row, col=reindex_col, value=batch)
            e_indices = batch
            u_indices = unique_row
            v_indices = unique_col

        # init result 初始化
        adjacencies = []
        e_flags = []

        ## for subsequent layers 对于每一层，进行多层的邻居采样
        for i in range(self.n_layers):

            # v -> u 基于u节点抽样新的v节点
            u_adj, next_v_indices = self.sample_v_given_u(
                u_indices, prev_v=v_indices, num_neighbors=self.num_neighbors_u[i]
            )
            # 构建从 v 节点到 u 节点的邻接矩阵
            dir_adj_v2u = self.adjacency_from_samples(
                u_adj, u_indices, next_v_indices, "v->u"
            )

            # u -> v
            v_adj, next_u_indices = self.sample_u_given_v(
                v_indices, prev_u=u_indices, num_neighbors=self.num_neighbors_v[i]
            )
            dir_adj_u2v = self.adjacency_from_samples(
                v_adj, next_u_indices, v_indices, "u->v"
            )

            # u -> e <- v 构建边连接的邻接矩阵
            dir_adj_e = self.adjacency_from_samples(
                e_adj, u_indices, v_indices, "u->e<-v"
            )

            # add them to the list
            adjacencies.append(BEANAdjacency(dir_adj_v2u, dir_adj_u2v, dir_adj_e))

            # for next iter
            e_adj, e_flag = self.combine_adjacency(
                v2u_adj=u_adj, u2v_adj=v_adj, e_adj=e_adj
            )
            u_indices = next_u_indices
            v_indices = next_v_indices
            e_flags.append(e_flag)

        # flip the order 调整数据的顺序，处理边标志
        # 如果只有一层，则取该层的邻接矩阵，否则反转列表
        # 当有多层时，采样的邻接矩阵列表会按照层级进行存储。但由于在反向传播时需要从最底层开始，
        # 所以需要将邻接矩阵的顺序反转，以便从最后一层到第一层进行反向传播。
        adjacencies = adjacencies[0] if len(adjacencies) == 1 else adjacencies[::-1]
        # 如果只有一个边标志，则取该标志，否则反转列表
        e_flags = e_flags[0] if len(e_flags) == 1 else e_flags[::-1]

        # get e_indices获取边的索引值
        e_indices = e_adj.storage.value()

        # print(f"sampling : {time.time() - start} s")
        # 添加负采样
        if self.neg_samples > 0:
            neg_samples = self.sample_negative(e_indices, self.neg_samples)
        else:
            neg_samples = None
        return batch_size, (u_indices, v_indices, e_indices), adjacencies, e_flags, neg_samples

# 专门用于加载图中的边数据，并进行采样
class EdgeLoader(torch.utils.data.DataLoader):
    def __init__(self, adj: SparseTensor, **kwargs):
        # 返回稀疏矩阵中的非零元素（即图中存在边的节点对）的数量。
        edge_idx = torch.arange(adj.nnz())
        self.adj = adj
        # 将 edge_idx 变成一维列表，表示所有边的索引
        super().__init__(edge_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    # 从批次 batch 中采样边，并生成一个子图的邻接矩阵。
    def sample(self, batch):

        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        # 当前批次中边的起始节点和目标节点的索引
        row = self.adj.storage.row()[batch]
        col = self.adj.storage.col()[batch]
        if self.adj.storage.has_value():
            val = self.adj.storage.col()[batch]
        else:
            val = batch

        # get unique row, col & idx
        # 由于某些u节点可能会在采样中重复出现 获取唯一的u节点索引，这样可以避免迭代计算
        unique_row, invidx_row = torch.unique(row, return_inverse=True)
        unique_col, invidx_col = torch.unique(col, return_inverse=True)

        # 图神经网络中节点的索引通常是稀疏且可能不连续的，但通过生成连续的索引
        reindex_row_id = torch.arange(len(unique_row))
        reindex_col_id = torch.arange(len(unique_col))

        # 重新编号应用到节点索引
        reindex_row = reindex_row_id[invidx_row]
        reindex_col = reindex_col_id[invidx_col]

        # 构造子图的邻接矩阵
        adj = SparseTensor(row=reindex_row, col=reindex_col, value=val)
        e_id = batch
        u_id = unique_row
        v_id = unique_col

        adj_e = DirectedAdj(adj, u_id, v_id, e_id, adj.sparse_sizes(), "u->e<-v")

        return adj_e
