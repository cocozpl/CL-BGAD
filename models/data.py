
import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_sparse.tensor import SparseTensor


class BipartiteData(Data):
    def __init__(
        self,
        adj: SparseTensor,
        xu: OptTensor = None,
        xv: OptTensor = None,
        xe: OptTensor = None,
        positive_pairs: OptTensor = None,  # 添加正样本对
        negative_pairs: OptTensor = None,  # 添加负样本对
        contrastive_labels: OptTensor = None,  # 对比学习标签
        **kwargs
    ):
        super().__init__()
        self.adj = adj
        self.xu = xu
        self.xv = xv
        self.xe = xe
        self.positive_pairs = positive_pairs  # 正样本对
        self.negative_pairs = negative_pairs  # 负样本对
        self.contrastive_labels = contrastive_labels  # 对比学习标签

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "adj":
            return torch.tensor([[self.xu.size(0)], [self.xv.size(0)]])
        elif key in ["positive_pairs", "negative_pairs"]:
            # 确保正负样本对的索引正确递增
            return self.xu.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
