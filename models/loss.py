
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor


def contrastive_loss(
        zu: Tensor,
        zv: Tensor,
        edge_pred_samples: SparseTensor,
        neg_samples: Tensor,
        tau: float = 0.5,
) -> Tensor:
    # Positive samples
    zu_edge = zu[edge_pred_samples.storage.row()]
    zv_edge = zv[edge_pred_samples.storage.col()]

    # Cosine similarity between positive samples
    pos_similarity = F.cosine_similarity(zu_edge, zv_edge)

    # Negative samples (without edges)
    neg_similarity = F.cosine_similarity(zu_edge.unsqueeze(1), neg_samples).mean(1)

    # InfoNCE loss calculation
    exp_pos = torch.exp(pos_similarity / tau)
    exp_neg = torch.exp(neg_similarity / tau)

    loss = -torch.log(exp_pos / (exp_pos + exp_neg))
    return loss.mean()


def reconstruction_loss(
        xu: Tensor,
        xv: Tensor,
        xe: Tensor,
        adj: SparseTensor,
        edge_pred_samples: SparseTensor,
        out: Dict[str, Tensor],
        xe_loss_weight: float = 1.0,
        structure_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        neg_samples: Tensor = None,
        tau: float = 0.5
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # feature mse
    xu_loss = F.mse_loss(xu, out["xu"])
    xv_loss = F.mse_loss(xv, out["xv"])
    xe_loss = F.mse_loss(xe, out["xe"])
    feature_loss = xu_loss + xv_loss + xe_loss_weight * xe_loss

    # structure loss
    edge_gt = (edge_pred_samples.storage.value() > 0).float()
    structure_loss = F.binary_cross_entropy(out["eprob"], edge_gt)

    # Contrastive loss
    if neg_samples is not None:
        contrast_loss = contrastive_loss(out["zu"], out["zv"], edge_pred_samples, neg_samples, tau)
    else:
        contrast_loss = 0.0

    # Total loss
    loss = feature_loss + structure_loss_weight * structure_loss + contrastive_loss_weight * contrast_loss

    loss_component = {
        "xu": xu_loss,
        "xv": xv_loss,
        "xe": xe_loss,
        "e": structure_loss,
        "contrastive": contrast_loss,
        "total": loss,
    }

    return loss, loss_component

