
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear

from typing import List, Optional, Tuple, Union, Dict

from models.conv_sample import BEANConvSample
from models.sampler import BEANAdjacency, BipartiteNeighborSampler, EdgeLoader
from utils.sparse_combine import xe_split3

from tqdm import tqdm

# 将输入 x 转换为一个元组 根据repeat参数，元组的长度可以是 2 或 3
def make_tuple(x: Union[int, Tuple[int, int, int], Tuple[int, int]], repeat: int = 3):
    if isinstance(x, int):
        if repeat == 2:
            return (x, x)
        else:
            return (x, x, x)
    else:
        return x


def apply_relu_dropout(x: Tensor, dropout_prob: float, training: bool) -> Tensor:
    x = F.relu(x)
    if dropout_prob > 0.0:
        x = F.dropout(x, p=dropout_prob, training=training)
    return x


class CL-BGADSampled(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int, int]],
        hidden_channels: Union[int, Tuple[int, int, int]] = 32,
        latent_channels: Union[int, Tuple[int, int]] = 64,
        edge_pred_latent: int = 64,
        n_layers_encoder: int = 4,
        n_layers_decoder: int = 4,
        n_layers_mlp: int = 4,
        dropout_prob: float = 0.0,
    ):

        super().__init__()

        self.in_channels = make_tuple(in_channels)
        self.hidden_channels = make_tuple(hidden_channels)
        self.latent_channels = make_tuple(latent_channels, 2)
        self.edge_pred_latent = edge_pred_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_mlp = n_layers_mlp
        self.dropout_prob = dropout_prob

        self.create_encoder()
        self.create_feature_decoder()
        self.create_structure_decoder()
    # 在encoder处需要加入节点的注意力机制
    def create_encoder(self):
        # self.encoder_convs 是一个 nn.ModuleList，用于存储网络中的所有编码器层（BEANConvSample）
        self.encoder_convs = nn.ModuleList()
        for i in range(self.n_layers_encoder):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_encoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.latent_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            if i == self.n_layers_encoder - 1:
                self.encoder_convs.append(
                    BEANConvSample(in_channels, out_channels, node_self_loop=False)
                )
            else:
                self.encoder_convs.append(
                    BEANConvSample(in_channels, out_channels, node_self_loop=True)
                )

    def create_feature_decoder(self):
        self.decoder_convs = nn.ModuleList()
        for i in range(self.n_layers_decoder):
            if i == 0:
                in_channels = self.latent_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_decoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.in_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            self.decoder_convs.append(BEANConvSample(in_channels, out_channels))
    # 在结构解码器这里添加对比学习，对比正负样本
    def create_structure_decoder(self):
        self.u_mlp_layers = nn.ModuleList()
        self.v_mlp_layers = nn.ModuleList()

        for i in range(self.n_layers_mlp):
            if i == 0:
                in_channels = self.latent_channels
            else:
                in_channels = (self.edge_pred_latent, self.edge_pred_latent)
            out_channels = self.edge_pred_latent

            self.u_mlp_layers.append(Linear(in_channels[0], out_channels))

            self.v_mlp_layers.append(Linear(in_channels[1], out_channels))

    def forward(
        self,
        xu: Tensor,
        xv: Tensor,
        xe: Tensor,
        bean_adjs: List[BEANAdjacency],
        e_flags: List[SparseTensor],
        edge_pred_samples: SparseTensor,
        neg_samples: Optional[SparseTensor] = None,
        contrastive_loss_weight: float = 1.0,
        tau: float = 0.5
    ) -> Dict[str, Tensor]:
        # 将 neg_samples 转换为标准的 Tensor (如果 neg_samples 存在)

        assert self.n_layers_encoder + self.n_layers_decoder == len(bean_adjs)

        # encoder
        for i, conv in enumerate(self.encoder_convs):
            badj = bean_adjs[i]
            e_flag = e_flags[i]

            n_ut = badj.adj_v2u.size[0]
            n_vt = badj.adj_u2v.size[1]

            xus, xut = xu, xu[:n_ut]
            xvs, xvt = xv, xv[:n_vt]
            if xe is not None:
                xe_e, xe_v2u, xe_u2v = xe_split3(xe, e_flag.storage.value())
            else:
                xe_e, xe_v2u, xe_u2v = None, None, None

            # xe_e, xe_v2u, xe_u2v = xe_split3(xe, e_flag.storage.value())

            xu, xv, xe = conv(
                xu=(xus, xut), xv=(xvs, xvt), adj=badj, xe=(xe_e, xe_v2u, xe_u2v)
            )

            if i != self.n_layers_encoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        last_badj = bean_adjs[-1]
        n_u_target = last_badj.adj_v2u.size[0]
        n_v_target = last_badj.adj_u2v.size[1]
        zu, zv = xu[:n_u_target], xv[:n_v_target]

        # feature decoder
        for i, conv in enumerate(self.decoder_convs):
            badj = bean_adjs[self.n_layers_encoder + i]
            e_flag = e_flags[self.n_layers_encoder + i]

            n_ut = badj.adj_v2u.size[0]
            n_vt = badj.adj_u2v.size[1]

            xus, xut = xu, xu[:n_ut]
            xvs, xvt = xv, xv[:n_vt]
            if xe is not None:
                xe_e, xe_v2u, xe_u2v = xe_split3(xe, e_flag.storage.value())
            else:
                xe_e, xe_v2u, xe_u2v = None, None, None

            xu, xv, xe = conv(
                xu=(xus, xut), xv=(xvs, xvt), adj=badj, xe=(xe_e, xe_v2u, xe_u2v)
            )

            if i != self.n_layers_decoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        # structure decoder
        # zu2, zv2 = zu, zv
        # # MLP for u
        # for i, layer in enumerate(self.u_mlp_layers):
        #     zu2 = layer(zu2)
        #     if i != self.n_layers_mlp - 1:
        #         zu2 = apply_relu_dropout(zu2, self.dropout_prob, self.training)
        #
        # # MLP for v
        # for i, layer in enumerate(self.v_mlp_layers):
        #     zv2 = layer(zv2)
        #     if i != self.n_layers_mlp - 1:
        #         zv2 = apply_relu_dropout(zv2, self.dropout_prob, self.training)
        #
        # # Select edges
        # zu2_edge = zu2[edge_pred_samples.storage.row()]
        # zv2_edge = zv2[edge_pred_samples.storage.col()]
        #
        # # Contrastive loss calculation (after MLP embedding)
        # contrastive_loss = 0
        # if neg_samples is not None:
        #     neg_samples_dense = neg_samples.to_dense()
        #
        #     # 计算正样本相似度
        #     pos_similarity = F.cosine_similarity(zu2_edge, zv2_edge)
        #
        #     # 计算需要重复的倍数，使负样本数量达到正样本数量
        #     repeat_factor = (zu2_edge.size(0) + neg_samples_dense.size(0) - 1) // neg_samples_dense.size(0)
        #
        #     # 使用 repeat 扩展负样本，使其数量与正样本数量匹配
        #     neg_samples_dense = neg_samples_dense.repeat(repeat_factor, 1)
        #
        #     # 负样本数量调整为与正样本数量一致
        #     neg_samples_dense = neg_samples_dense[:zu2_edge.size(0)]
        #
        #     # 确保负样本的特征维度（最后一个维度）与正样本一致
        #     if zu2_edge.shape[-1] != neg_samples_dense.shape[-1]:
        #         neg_samples_dense = neg_samples_dense[..., :zu2_edge.shape[-1]]
        #
        #     # 计算负样本相似度
        #     neg_similarity = F.cosine_similarity(zu2_edge.unsqueeze(1), neg_samples_dense.unsqueeze(0), dim=2)
        #
        #     # InfoNCE Loss 计算
        #     temperature = tau  # 使用原有的 tau 作为温度参数
        #     pos_similarity = pos_similarity.unsqueeze(1)  # 调整维度以匹配 neg_similarity
        #     logits = torch.cat([pos_similarity, neg_similarity], dim=1) / temperature
        #     labels = torch.zeros(zu2_edge.size(0), dtype=torch.long, device=zu2_edge.device)
        #     contrastive_loss = F.cross_entropy(logits, labels)
        #
        # # After contrastive loss, calculate the final edge probability using sigmoid
        # eprob = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))
        # structure decoder
        zu2, zv2 = zu, zv
        for i, layer in enumerate(self.u_mlp_layers):
            zu2 = layer(zu2)
            if i != self.n_layers_mlp - 1:
                zu2 = apply_relu_dropout(zu2, self.dropout_prob, self.training)

        for i, layer in enumerate(self.v_mlp_layers):
            zv2 = layer(zv2)
            if i != self.n_layers_mlp - 1:
                zv2 = apply_relu_dropout(zv2, self.dropout_prob, self.training)

        zu2_edge = zu2[edge_pred_samples.storage.row()]
        zv2_edge = zv2[edge_pred_samples.storage.col()]
        # eprob = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))

        # contrastive loss
        contrastive_loss = 0
        if neg_samples is not None:
            neg_samples_dense = neg_samples.to_dense()

            # print(f"zu2_edge shape: {zu2_edge.shape}")
            # print(f"neg_samples_dense shape: {neg_samples_dense.shape}")

            # 计算正样本相似度
            pos_similarity = F.cosine_similarity(zu2_edge, zv2_edge)

            # 计算需要重复的倍数，使负样本数量达到正样本数量
            repeat_factor = (zu2_edge.size(0) + neg_samples_dense.size(0) - 1) // neg_samples_dense.size(0)

            # 使用 repeat 扩展负样本，使其数量与正样本数量匹配
            neg_samples_dense = neg_samples_dense.repeat(repeat_factor, 1)

            # 负样本数量调整为与正样本数量一致
            neg_samples_dense = neg_samples_dense[:zu2_edge.size(0)]

            # 确保负样本的特征维度（最后一个维度）与正样本一致
            if zu2_edge.shape[-1] != neg_samples_dense.shape[-1]:
                neg_samples_dense = neg_samples_dense[..., :zu2_edge.shape[-1]]

            # print(f"Adjusted neg_samples_dense shape: {neg_samples_dense.shape}")

            # 直接计算负样本相似度
            neg_similarity = F.cosine_similarity(zu2_edge, neg_samples_dense, dim=-1)

            # 计算对比损失
            exp_pos = torch.exp(pos_similarity / tau)
            exp_neg = torch.exp(neg_similarity / tau)
            contrastive_loss = -torch.log(exp_pos / (exp_pos + exp_neg)).mean()

        eprob = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))
            # print(f"Contrastive loss: {contrastive_loss}")

        # collect results
        result = {
            "xu": xu,
            "xv": xv,
            "xe": xe,
            "zu": zu,
            "zv": zv,
            "eprob": eprob,
            "contrastive_loss": contrastive_loss_weight * contrastive_loss
        }

        return result

    # 执行单次卷积操作
    def apply_conv(self, conv, dir_adj, xu_all, xv_all, xe_all, device):
        # dir_adj：包含方向的邻接矩阵，adj_v2u或adj_u2v
        xu = xu_all[dir_adj.u_id].to(device)
        xv = xv_all[dir_adj.v_id].to(device)
        xe = xe_all[dir_adj.e_id].to(device) if xe_all is not None else None
        adj = dir_adj.adj.to(device)

        out = conv((xu, xv), adj, xe)

        return out
    # 执行整个推理过程，包括加载批量数据，应用卷积操作
    def inference(
        self,
        xu_all: Tensor,
        xv_all: Tensor,
        xe_all: Tensor,
        adj_all: SparseTensor,
        edge_pred_samples: SparseTensor,
        batch_sizes: Tuple[int, int, int],
        device,
        progress_bar: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        # 禁用数据的随机打乱，在推理过程中通常不需要打乱数据。
        kwargs["shuffle"] = False
        u_loader = BipartiteNeighborSampler(
            adj_all,
            n_layers=1,
            base="u",
            # batch_sizes[0]：表示在采样 u 节点时，每次加载的数据的批量大小。
            batch_size=batch_sizes[0],
            n_other_node=1,
            num_neighbors_u=-1, # 表示采样 u 节点的所有邻居
            num_neighbors_v=1, # 采样 v 节点的 1 个邻居
            **kwargs,
        )
        v_loader = BipartiteNeighborSampler(
            adj_all,
            n_layers=1,
            base="v",
            batch_size=batch_sizes[1],
            n_other_node=1,
            num_neighbors_u=1,
            num_neighbors_v=-1,
            **kwargs,
        )
        e_loader = EdgeLoader(adj_all, batch_size=batch_sizes[2], **kwargs)

        u_mlp_loader = torch.utils.data.DataLoader(
            torch.arange(xu_all.shape[0]), batch_size=batch_sizes[0], **kwargs
        )
        v_mlp_loader = torch.utils.data.DataLoader(
            torch.arange(xv_all.shape[0]), batch_size=batch_sizes[1], **kwargs
        )
        # 加载需要预测的边样本 edge_pred_samples.nnz()返回非零样本的数量 nnz表示"non-zero"，这是稀疏矩阵中非零元素的数量
        epred_loader = torch.utils.data.DataLoader(
            torch.arange(edge_pred_samples.nnz()), batch_size=batch_sizes[2], **kwargs
        )
        # 计算推理过程中需要的总迭代次数
        total_iter = (
            (len(u_loader) + len(v_loader))
            * (self.n_layers_encoder + self.n_layers_decoder)
            # 边特征的卷积通常在每层进行，但最后一层可能不需要对边特征进行处理
            + len(e_loader) * (self.n_layers_encoder + self.n_layers_decoder - 1)
            + (len(u_mlp_loader) + len(v_mlp_loader)) * self.n_layers_mlp
            + len(epred_loader)
        )
        if progress_bar:
            pbar = tqdm(total=total_iter, leave=False)
            pbar.set_description(f"Evaluation")

        # encoder  加节点的注意力机制
        # 前向传播逻辑
        # self.encoder_convs：存储了所有编码器层（卷积层）。遍历每一个编码器层
        for i, conv in enumerate(self.encoder_convs):
            ## next u nodes
            # 用于存储本次迭代中处理u类型节点后得到的输出结果。每次从u_loader中采样的批次结果会被存储到这个列表中，
            # 最后使用torch.cat将它们连接起来
            xu_list = []
            for _, _, adjacency, _, neg_samples in u_loader:
                # 在这里处理 adjacency 和 neg_samples

                out = self.apply_conv(
                    conv.v2u_conv, adjacency.adj_v2u, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_encoder - 1:
                    out = F.relu(out)
                xu_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xu_all_next = torch.cat(xu_list, dim=0)

            ## next v nodes
            xv_list = []
            # _ 是 Python 中常见的占位符 表示我们不关心这些变量的值
            for _, _, adjacency, _, neg_samples in v_loader:

                # 当前卷积层用于从 u 节点向 v 节点传递信息的卷积部分
                out = self.apply_conv(
                    conv.u2v_conv, adjacency.adj_u2v, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_encoder - 1:
                    out = F.relu(out)
                xv_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xv_all_next = torch.cat(xv_list, dim=0)

            ## next edge
            if i != self.n_layers_encoder - 1:
                xe_list = []
                for adj_e in e_loader:
                    out = self.apply_conv(
                        conv.e_conv, adj_e, xu_all, xv_all, xe_all, device
                    )
                    out = F.relu(out)
                    xe_list.append(out.cpu())
                    if progress_bar:
                        pbar.update(1)
                xe_all_next = torch.cat(xe_list, dim=0)
            else:
                xe_all_next = None

            xu_all = xu_all_next
            xv_all = xv_all_next
            xe_all = xe_all_next

        # get latent vars
        zu_all, zv_all = xu_all, xv_all

        # feature decoder
        for i, conv in enumerate(self.decoder_convs):

            ## next u nodes
            xu_list = []
            for _, _, adjacency, _, neg_samples in u_loader:
                # 在这里处理 adjacency 和 neg_samples

                out = self.apply_conv(
                    conv.v2u_conv, adjacency.adj_v2u, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xu_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xu_all_next = torch.cat(xu_list, dim=0)

            ## next v nodes
            xv_list = []
            for _, _, adjacency, _, neg_samples in v_loader:

                out = self.apply_conv(
                    conv.u2v_conv, adjacency.adj_u2v, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xv_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xv_all_next = torch.cat(xv_list, dim=0)

            ## next edge
            xe_list = []
            for adj_e in e_loader:
                out = self.apply_conv(
                    conv.e_conv, adj_e, xu_all, xv_all, xe_all, device
                )
                if i != self.n_layers_decoder - 1:
                    out = F.relu(out)
                xe_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            xe_all_next = torch.cat(xe_list, dim=0)

            xu_all = xu_all_next
            xv_all = xv_all_next
            xe_all = xe_all_next

        # structure decoder 要加上对比学习
        # zu2_all, zv2_all：分别用于存储u和v节点的特征，这些特征会经过MLP层进一步处理
        zu2_all, zv2_all = zu_all, zv_all
        # i：当前 MLP 层的索引 layer：表示当前的 MLP 层
        for i, layer in enumerate(self.u_mlp_layers):
            zu2_list = []
            for batch in u_mlp_loader:
                out = layer(zu2_all[batch].to(device))
                if i != self.n_layers_mlp - 1:
                    out = F.relu(out)
                zu2_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            zu2_all = torch.cat(zu2_list, dim=0)

        for i, layer in enumerate(self.v_mlp_layers):
            zv2_list = []
            for batch in v_mlp_loader:
                out = layer(zv2_all[batch].to(device))
                if i != self.n_layers_mlp - 1:
                    out = F.relu(out)
                zv2_list.append(out.cpu())
                if progress_bar:
                    pbar.update(1)
            zv2_all = torch.cat(zv2_list, dim=0)

        eprob_list = []
        for batch in epred_loader:
            zu2_edge = zu2_all[edge_pred_samples.storage.row()[batch]].to(device)
            zv2_edge = zv2_all[edge_pred_samples.storage.col()[batch]].to(device)
            out = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))
            eprob_list.append(out.cpu())
            if progress_bar:
                pbar.update(1)
        eprob_all = torch.cat(eprob_list, dim=0)

        # collect results
        result = {
            "xu": xu_all,
            "xv": xv_all,
            "xe": xe_all,
            "zu": zu_all,
            "zv": zv_all,
            "eprob": eprob_all,
            # "contrastive_loss": neg_similarity  # 返回对比学习的负样本相似度

        }

        if progress_bar:
            pbar.close()

        return result
