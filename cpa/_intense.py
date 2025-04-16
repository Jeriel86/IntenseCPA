from typing import Union, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

from ._normalization_module import VectorWiseBatchNorm, Normalize3


class MKLFusion(nn.Module):
    def __init__(
            self,
            in_features: dict[str, int],
            out_features: int,
            bias: bool = True,
            reg_rate: float = 0.01,
            intense_p: int = 1
    ):
        """Initialize the MKLFusion module
        Args:
            in_features(dict[str, int]): A dictionary where keys are the modality keys and
                                          values are their respective feature dimensions.
            out_features(int): Final output dimension (e.g., number of classes).
            bias(bool): Whether to use a bias in the fusion layer.
            reg_rate(float): Regularization rate.
            intense_p(int): Hyperparameter p used for the intensity regularization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fusion_layer = nn.Linear(
            in_features=sum(in_features.values()),
            out_features=out_features,
            bias=bias
        )
        self.reg_rate = reg_rate
        self.p = intense_p

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

    def regularizer(self):
        q = 2 * self.p / (self.p + 1)
        return self.reg_rate * torch.sum(self.weight_norms() ** q) ** (2 / q)

    def scores(self):
        with torch.no_grad():
            norms = self.weight_norms()
            a = norms ** (2 / (self.p + 1))
            b = torch.sum(norms ** (2 * self.p / (self.p + 1))) ** (1 / self.p)
            scores = a / b
            scores = scores / torch.sum(scores)
            scores = scores.cpu().numpy()
            return dict(zip(self.in_features.keys(), scores))

    def weight_norms(self):
        return torch.stack(
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
            raise ValueError('Tensor product is only supported for 2 to 4 batch vectors.')


from typing import Union, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

from ._normalization_module import VectorWiseBatchNorm, Normalize3

class MKLFusion(nn.Module):
    # (Original MKLFusion code; unchanged)
    def __init__(self,
                 in_features: dict[str, int],
                 out_features: int,
                 bias: bool = True,
                 reg_rate: float = 0.01,
                 intense_p: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fusion_layer = nn.Linear(
            in_features=sum(in_features.values()),
            out_features=out_features,
            bias=bias
        )
        self.reg_rate = reg_rate
        self.p = intense_p

    @property
    def bias(self):
        return self.fusion_layer.bias

    @property
    def weight(self):
        return torch.split(self.fusion_layer.weight,
                           list(self.in_features.values()),
                           dim=1)

    def forward(self, tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
        tensors = torch.cat(tensors, dim=1)
        return self.fusion_layer(tensors)

    def regularizer(self) -> Tensor:
        q = 2 * self.p / (self.p + 1)
        return self.reg_rate * torch.sum(self.weight_norms() ** q) ** (2 / q)

    def scores(self) -> dict:
        with torch.no_grad():
            norms = self.weight_norms()
            a = norms ** (2 / (self.p + 1))
            b = torch.sum(norms ** (2 * self.p / (self.p + 1))) ** (1 / self.p)
            scores = a / b
            scores = scores / torch.sum(scores)
            scores = scores.numpy()
            return dict(zip(self.in_features.keys(), scores))

    def weight_norms(self) -> Tensor:
        return torch.tensor(
            [torch.linalg.matrix_norm(tens) for tens in self.weight]
        )


class TensorFusionModel(nn.Module):
    # (Legacy tensor fusion model – provided for fallback)
    def __init__(self, modality_indices: List[str], input_dim: int = 16) -> None:
        super().__init__()
        self.modality_indices = modality_indices
        self.input_dim = input_dim

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        tens_list = [x[index] for index in self.modality_indices]
        t: Tensor = self.compute_tensor_product(tens_list)
        t = torch.flatten(t, start_dim=1)
        return t

    def compute_tensor_product(self, inp: List[Tensor]) -> Tensor:
        if len(inp) == 2:
            return torch.einsum("bi,bj->bij", inp)
        elif len(inp) == 3:
            return torch.einsum("bi,bj,bk->bijk", inp)
        elif len(inp) == 4:
            return torch.einsum("bi,bj,bk,bl->bijkl", inp)
        else:
            raise ValueError('Tensor product is only supported for 2 to 4 batch vectors.')


class InTense(nn.Module):
    """
    InTense module with configurable interaction order and low-rank factorization.
    It applies batch normalization on single embeddings, and computes pairwise and triple
    interactions using low-rank projections. Additionally, for triple interactions, it uses
    extra projections (single_out_proj) to map each modality’s low-rank vector to the common
    output space for proper normalization.
    """
    def __init__(
        self,
        dim_dict_single: dict[str, int],
        feature_dim_dict_triple: dict[str, int],
        track_running_stats: bool = True,
        out_features: int = 1,
        intense_reg_rate: float = 0.01,
        intense_p: int = 1,
        interaction_order: int = 3,  # 1: singles; 2: singles + pairs; 3: singles + pairs + triple
        intense_interaction_rank: int = 64  # Low-rank projection size for interactions
    ):
        super().__init__()
        if interaction_order not in [1, 2, 3]:
            raise ValueError("interaction_order must be 1, 2, or 3")
        self.interaction_order = interaction_order
        self.intense_interaction_rank = intense_interaction_rank

        # Assume modalities "1", "2", "3" all have dimension n_latent
        common_single_dim = None
        if dim_dict_single["1"] == dim_dict_single["2"] == dim_dict_single.get("3", dim_dict_single["1"]):
            common_single_dim = dim_dict_single["1"]

        # If low-rank factorization is active, we set interaction outputs to common_single_dim.
        if intense_interaction_rank > 0:
            if common_single_dim is not None:
                interaction_out_dim = common_single_dim
            else:
                interaction_out_dim = intense_interaction_rank

            feature_dim_dict_triple["12"] = interaction_out_dim
            feature_dim_dict_triple["13"] = interaction_out_dim
            feature_dim_dict_triple["23"] = interaction_out_dim
            if self.interaction_order == 3:
                feature_dim_dict_triple["123"] = interaction_out_dim

        # (A) Batch normalization for single-modality embeddings.
        self.bn_single = nn.ModuleDict({
            mod_idx: VectorWiseBatchNorm(
                num_features=dim_dict_single[mod_idx],
                track_running_stats=track_running_stats
            ) for mod_idx in ["1", "2", "3"]
        })

        # (B) Low-rank projections for interactions.
        self.proj_layers = None
        self.out_proj_pair = None
        self.out_proj_triple = None
        if self.interaction_order >= 2 and intense_interaction_rank > 0:
            self.proj_layers = nn.ModuleDict({
                mod: nn.Linear(dim_dict_single[mod], intense_interaction_rank)
                for mod in ["1", "2", "3"]
            })
            # Optional projection to map interaction features back to common_single_dim:
            if common_single_dim is not None and intense_interaction_rank != common_single_dim:
                self.out_proj_pair = nn.ModuleDict({
                    "12": nn.Linear(intense_interaction_rank, interaction_out_dim),
                    "13": nn.Linear(intense_interaction_rank, interaction_out_dim),
                    "23": nn.Linear(intense_interaction_rank, interaction_out_dim)
                })
                if self.interaction_order == 3:
                    self.out_proj_triple = nn.Linear(intense_interaction_rank, interaction_out_dim)
            else:
                self.out_proj_pair = None
                self.out_proj_triple = None

            # NEW: Single modality output projections to map each projected vector to common_single_dim.
            if common_single_dim is not None and intense_interaction_rank != common_single_dim:
                self.single_out_proj = nn.ModuleDict({
                    "1": nn.Linear(intense_interaction_rank, common_single_dim),
                    "2": nn.Linear(intense_interaction_rank, common_single_dim),
                    "3": nn.Linear(intense_interaction_rank, common_single_dim)
                })
            else:
                self.single_out_proj = None

        # (C) If not using low-rank, fall back to legacy outer-product method.
        if self.interaction_order >= 2 and self.intense_interaction_rank <= 0:
            self.tf_12 = TensorFusionModel(["1", "2"])
            self.tf_13 = TensorFusionModel(["1", "3"])
            self.tf_23 = TensorFusionModel(["2", "3"])

        # (D) Batch normalization for pairwise interactions.
        if self.interaction_order >= 2:
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

        # (E) Triple interaction module.
        if self.interaction_order == 3:
            if self.intense_interaction_rank <= 0:
                self.tf_123 = TensorFusionModel(["1", "2", "3"])
            self.normalize_triple = Normalize3(
                feature_dim_dict=feature_dim_dict_triple,
                track_running_stats=track_running_stats,
                mod_index="123"
            )

        # (F) MKL Fusion: Prepare dictionary for fusion using keys for singles and interactions.
        in_feats = {"z1": dim_dict_single["1"],
                    "z2": dim_dict_single["2"],
                    "z3": dim_dict_single["3"]}
        if self.interaction_order >= 2:
            in_feats["z12"] = feature_dim_dict_triple["12"]
            in_feats["z13"] = feature_dim_dict_triple["13"]
            in_feats["z23"] = feature_dim_dict_triple["23"]
        if self.interaction_order == 3:
            in_feats["z123"] = feature_dim_dict_triple["123"]

        self.mkl_fusion = MKLFusion(
            in_features=in_feats,
            out_features=out_features,
            bias=True,
            reg_rate=intense_reg_rate,
            intense_p=intense_p
        )

    def forward(self, z_dict: dict[str, Tensor]) -> Tensor:
        # (1) Normalize single modalities.
        z1 = self.bn_single["1"](z_dict["1"])
        z2 = self.bn_single["2"](z_dict["2"])
        z3 = self.bn_single["3"](z_dict["3"])
        z_dict_norm = {"1": z1, "2": z2, "3": z3}

        tensor_list: List[Tensor] = [z1, z2, z3]

        # (2) Pairwise interactions.
        if self.interaction_order >= 2:
            if self.intense_interaction_rank > 0 and self.proj_layers is not None:
                # Project each modality.
                z1_proj = self.proj_layers["1"](z_dict_norm["1"])  # [B, rank]
                z2_proj = self.proj_layers["2"](z_dict_norm["2"])  # [B, rank]
                z3_proj = self.proj_layers["3"](z_dict_norm["3"])  # [B, rank]

                # Compute elementwise product.
                z12 = z1_proj * z2_proj  # [B, rank]
                z13 = z1_proj * z3_proj
                z23 = z2_proj * z3_proj

                # Map pair interactions back to desired output dimension, if applicable.
                if self.out_proj_pair is not None:
                    z12 = self.out_proj_pair["12"](z12)
                    z13 = self.out_proj_pair["13"](z13)
                    z23 = self.out_proj_pair["23"](z23)

                # Apply BN.
                z12 = self.bn_pair["12"](z12)
                z13 = self.bn_pair["13"](z13)
                z23 = self.bn_pair["23"](z23)
            else:
                z12 = self.bn_pair["12"]( self.tf_12(z_dict_norm) )
                z13 = self.bn_pair["13"]( self.tf_13(z_dict_norm) )
                z23 = self.bn_pair["23"]( self.tf_23(z_dict_norm) )

            tensor_list.extend([z12, z13, z23])
        else:
            z12 = z13 = z23 = None

        # (3) Triple interaction.
        if self.interaction_order == 3:
            if self.intense_interaction_rank > 0 and self.proj_layers is not None:
                # Reuse projected vectors; assume z1_proj, z2_proj, z3_proj are available.
                # (If not computed in pairwise block, compute them here.)
                try:
                    z1_proj
                except NameError:
                    z1_proj = self.proj_layers["1"](z_dict_norm["1"])
                    z2_proj = self.proj_layers["2"](z_dict_norm["2"])
                    z3_proj = self.proj_layers["3"](z_dict_norm["3"])
                z123 = z1_proj * z2_proj * z3_proj  # [B, rank]
                if self.out_proj_triple is not None:
                    z123 = self.out_proj_triple(z123)

                # Compute means of pairwise interactions (already in output dimension).
                mean_12 = torch.mean(z12, dim=0)  # [output_dim]
                mean_13 = torch.mean(z13, dim=0)
                mean_23 = torch.mean(z23, dim=0)
                mean_123 = torch.mean(z123, dim=0)

                # Transform modality projections to output dimension before using in triple normalization.
                t_z1 = self.single_out_proj["1"](z1_proj) if self.single_out_proj is not None else z1_proj
                t_z2 = self.single_out_proj["2"](z2_proj) if self.single_out_proj is not None else z2_proj
                t_z3 = self.single_out_proj["3"](z3_proj) if self.single_out_proj is not None else z3_proj

                out_12_3 = mean_12 * t_z3
                out_13_2 = mean_13 * t_z2
                out_23_1 = mean_23 * t_z1
                predicted_triple = mean_123 + out_12_3 + out_13_2 + out_23_1
                diff = z123 - predicted_triple
                var = torch.sum(diff * diff, dim=0) / z123.shape[0]
                z123_norm = (z123 - predicted_triple) / torch.sqrt(var + self.normalize_triple.eps)

                # Update running statistics if needed.
                if self.normalize_triple.track_running_stats:
                    with torch.no_grad():
                        m = self.normalize_triple.momentum
                        if m is None:
                            m = 1.0 / float(self.normalize_triple.num_batches_tracked + 1) if self.normalize_triple.num_batches_tracked is not None else 0.0
                        if self.normalize_triple.num_batches_tracked is not None:
                            self.normalize_triple.num_batches_tracked += 1
                        self.normalize_triple.running_mean = (m * mean_123 + (1 - m) * self.normalize_triple.running_mean) if self.normalize_triple.running_mean is not None else mean_123
                        self.normalize_triple.running_mean_12 = (m * mean_12 + (1 - m) * self.normalize_triple.running_mean_12) if self.normalize_triple.running_mean_12 is not None else mean_12
                        self.normalize_triple.running_mean_13 = (m * mean_13 + (1 - m) * self.normalize_triple.running_mean_13) if self.normalize_triple.running_mean_13 is not None else mean_13
                        self.normalize_triple.running_mean_23 = (m * mean_23 + (1 - m) * self.normalize_triple.running_mean_23) if self.normalize_triple.running_mean_23 is not None else mean_23
                        unbiased_var = var * (z123.shape[0] / max(z123.shape[0] - 1, 1))
                        self.normalize_triple.running_var = (m * unbiased_var + (1 - m) * self.normalize_triple.running_var) if self.normalize_triple.running_var is not None else unbiased_var

                tensor_list.append(z123_norm)
            else:
                z123 = self.tf_123(z_dict_norm)
                input_dict = {"12": z12, "13": z13, "23": z23, "123": z123}
                pre_fusion_dict = {"1": z1, "2": z2, "3": z3}
                z123_norm = self.normalize_triple(input_dict, pre_fusion_dict)
                tensor_list.append(z123_norm)

        # (4) Fuse all representations using MKLFusion.
        final_out = self.mkl_fusion(tensor_list)
        return final_out

    def get_regularizer(self) -> Tensor:
        return self.mkl_fusion.regularizer()

    def get_relevance_score(self) -> dict:
        return self.mkl_fusion.scores()
