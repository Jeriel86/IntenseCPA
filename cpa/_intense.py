from typing import Union, Tuple, List
import logging
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math

from ._normalization_module import VectorWiseBatchNorm, Normalize3


class MKLFusion(nn.Module):
    def __init__(
            self,
            # in_tensors: int,
            in_features: dict[str, int],
            out_features: int,
            bias: bool = True,
            reg_rate: float = 0.01
    ):
        """Initialize the MKLFusion module
        Args:
            in_features(list[int]): Each element of the list is the dimension of a
                                    modality's feature-representation, which is an input to
                                    the final fusion layer
            out_features(int): Final output which depends on the number if classes
                                in the dataset
        Returns:
            A nn.Module to fuse the mdalities
        """
        # TODO: add an option to pass the hyperparameter p here
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fusion_layer = nn.Linear(
            in_features=sum(in_features.values()),
            out_features=out_features,
            bias=bias
        )
        self.reg_rate = reg_rate


    @property
    def bias(self):
        return self.fusion_layer.bias

    @property
    def weight(self):
        return torch.split(self.fusion_layer.weight,
                           list(self.in_features.values()),
                           dim=1)

    def forward(self, tensors: Union[Tuple[torch.Tensor, ...], List[Tensor]]):
        tensors = torch.cat(tensors, dim=1)
        return self.fusion_layer(tensors)

    def regularizer(self, p=1):
        q = 2 * p / (p + 1)
        return self.reg_rate * torch.sum(self.weight_norms() ** q) ** (2 / q)

    def scores(self, p=1):
        with torch.no_grad():
            norms = self.weight_norms()
            a = norms ** (2 / (p + 1))
            b = torch.sum(norms ** (2 * p / (p + 1))) ** (1 / p)
            scores = a / b
            scores = scores.numpy()
            return dict(zip(self.in_features.keys(), scores))

    def weight_norms(self):
        return torch.tensor(
            [torch.linalg.matrix_norm(tens) for tens in self.weight]
        )


class TensorFusionModel(nn.Module):
    def __init__(self, modality_indices: list[str],
                 input_dim: int = 16) -> None:
        super().__init__()
        self.modality_indices = modality_indices
        self.input_dim = input_dim

    def forward(self, x: dict[str, torch.Tensor]):
        tens_list_x = [x[index] for index in self.modality_indices]
        t: torch.Tensor = self.compute_tensor_product(tens_list_x)
        t = torch.flatten(t, start_dim=1)
        return t

    def compute_tensor_product(self, inp: list[torch.Tensor]) -> torch.Tensor:
        if len(inp) == 2:
            return torch.einsum("bi,bj->bij", inp)
        elif len(inp) == 3:
            return torch.einsum("bi,bj,bk->bijk", inp)
        elif len(inp) == 4:
            return torch.einsum("bi,bj,bk,bl->bijkl", inp)
        else:
            raise ValueError('Tensor product is only supported for 2 to 4\
                 batch vectors.')



class InTense(nn.Module):
    """
    Full pipeline:
      - BN singles
      - BN pairs
      - triple normalization
      - fuse all (singles + pairs + triple) with MKLFusion
    """

    def __init__(
        self,
        dim_dict_single: dict[str, int],
        feature_dim_dict_triple: dict[str, int],
        track_running_stats: bool = True,
        out_features: int = 1,
        intense_reg_rate: float = 0.01
    ):
        super().__init__()

        # (A) BN on single embeddings
        self.bn_single = nn.ModuleDict({
            mod_idx: VectorWiseBatchNorm(
                num_features=dim_dict_single[mod_idx],
                track_running_stats=track_running_stats
            )
            for mod_idx in ["1", "2", "3"]
        })

        # (B) Models for pairwise & triple
        self.tf_12 = TensorFusionModel(["1", "2"])
        self.tf_13 = TensorFusionModel(["1", "3"])
        self.tf_23 = TensorFusionModel(["2", "3"])
        self.tf_123 = TensorFusionModel(["1", "2", "3"])

        # (C) BN on each pairwise product
        self.bn_pair = nn.ModuleDict({
            "12": VectorWiseBatchNorm(
                num_features=feature_dim_dict_triple["12"],
                track_running_stats=track_running_stats
            ),
            "13": VectorWiseBatchNorm(
                num_features=feature_dim_dict_triple["13"],
                track_running_stats=track_running_stats
            ),
            "23": VectorWiseBatchNorm(
                num_features=feature_dim_dict_triple["23"],
                track_running_stats=track_running_stats
            ),
        })

        # (D) Normalization for triple
        self.normalize_triple = Normalize3(
            feature_dim_dict=feature_dim_dict_triple,
            track_running_stats=track_running_stats,
            mod_index="123"
        )

        # (E) Finally, an MKLFusion layer to fuse ALL representations:
        #     We'll give it a dictionary of in_features,
        #     summing = z1_dim + z2_dim + z3_dim + z12_dim + z13_dim + z23_dim + z123_dim
        in_feats = {
            "z1":  dim_dict_single["1"],
            "z2":  dim_dict_single["2"],
            "z3":  dim_dict_single["3"],
            "z12": feature_dim_dict_triple["12"],
            "z13": feature_dim_dict_triple["13"],
            "z23": feature_dim_dict_triple["23"],
            "z123": feature_dim_dict_triple["123"],
        }
        self.intense_reg_rate = intense_reg_rate
        self.mkl_fusion = MKLFusion(
            in_features=in_feats,
            out_features=out_features,
            bias=True,
            reg_rate=intense_reg_rate
        )

    def forward(self, z_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        z_dict = {
          "1": [B, dim1], "2": [B, dim2], "3": [B, dim3]
        }
        Returns:
          final_out: [B, out_features], e.g. classification logits or regression output
        """

        # 1) BN on single embeddings
        z1 = self.bn_single["1"](z_dict["1"])
        z2 = self.bn_single["2"](z_dict["2"])
        z3 = self.bn_single["3"](z_dict["3"])

        # pack them
        z_dict_norm = {"1": z1, "2": z2, "3": z3}

        # 2) Pairwise + BN
        z12 = self.tf_12(z_dict_norm)  # shape [B, dim12]
        z13 = self.tf_13(z_dict_norm)
        z23 = self.tf_23(z_dict_norm)

        z12 = self.bn_pair["12"](z12)
        z13 = self.bn_pair["13"](z13)
        z23 = self.bn_pair["23"](z23)

        # 3) Triple product + specialized BN
        z123 = self.tf_123(z_dict_norm)
        input_dict = {"12": z12, "13": z13, "23": z23, "123": z123}
        pre_fusion_dict = {"1": z1, "2": z2, "3": z3}
        z123_norm = self.normalize_triple(input_dict, pre_fusion_dict)

        # 4) Pass everything to MKLFusion
        #    combine singles, pairs, triple, e.g.:
        all_tensors = [z1, z2, z3, z12, z13, z23, z123_norm]

        final_out = self.mkl_fusion(all_tensors)  # shape [B, out_features]
        #print(self.get_relevance_score())

        return final_out

    def get_regularizer(self):
        return self.mkl_fusion.regularizer()

    def get_relevance_score(self):
        return self.mkl_fusion.scores()

