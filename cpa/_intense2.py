from __future__ import annotations

"""InTense components: MKLFusion, TensorFusionModel, and the main InTense module.

This version adds a **tf_latent_dim** hyper‑parameter that linearly projects each
single‑modality representation to a fixed dimensionality *before* any tensor
products are taken, exactly as described in Appendix E.5 of the paper (see the
row labelled “tf latent dim”).
"""

from typing import Union, Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ._normalization_module import VectorWiseBatchNorm, Normalize3


# -----------------------------------------------------------------------------
#                                MKL‑Fusion
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
#                             Tensor‑product helper
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
#                                   InTense
# -----------------------------------------------------------------------------
class InTense(nn.Module):
    """
    Full pipeline with configurable interaction order:
      - BN on single embeddings (order 1)
      - **Optional** linear projection to *tf_latent_dim*.
      - Optionally, pairwise interactions (order 2)
      - Optionally, triple interaction (order 3)
      - Fuse all selected representations with MKLFusion
    """

    def __init__(
        self,
        dim_dict_single: dict[str, int],
        feature_dim_dict_triple: dict[str, int],
        track_running_stats: bool = True,
        out_features: int = 1,
        intense_reg_rate: float = 0.01,
        intense_p: int = 1,
        interaction_order: int = 3,  # 1: singles, 2: singles + pairs, 3: singles + pairs + triple
        tf_latent_dim: Optional[int] = None,
    ):
        """
        Args:
            dim_dict_single(dict[str, int]): Dictionary with dimensions of single-modality features.
            feature_dim_dict_triple(dict[str, int]): Dictionary with dimensions for pairwise and triple interactions.
            track_running_stats(bool): Whether BN modules track running stats.
            out_features(int): Output dimension.
            intense_reg_rate(float): Regularization rate for MKLFusion.
            intense_p(int): Hyperparameter p for MKLFusion.
            interaction_order(int): Interaction order to use (1, 2, or 3).
        """
        super().__init__()

        if interaction_order not in [1, 2, 3]:  # sanity check
            raise ValueError("interaction_order must be 1, 2, or 3")
        self.interaction_order = interaction_order
        self.tf_latent_dim = tf_latent_dim

        # ------------------------------------------------------------------
        # (A)   VBN on single‑modality embeddings
        # ------------------------------------------------------------------
        self.bn_single = nn.ModuleDict({
            mod_idx: VectorWiseBatchNorm(
                num_features=dim_dict_single[mod_idx],
                track_running_stats=track_running_stats
            )
            for mod_idx in ["1", "2", "3"]
        })

        # (A.1) *Optional* linear projection → tf_latent_dim ----------------
        if self.tf_latent_dim is not None:
            self.proj_single = nn.ModuleDict({
                mod_idx: nn.Linear(dim_dict_single[mod_idx], self.tf_latent_dim)
                for mod_idx in ["1", "2", "3"]
            })

        # ------------------------------------------------------------------
        # (B)   Pairwise interactions
        # ------------------------------------------------------------------
        if self.interaction_order >= 2:
            self.tf_12 = TensorFusionModel(["1", "2"])
            self.tf_13 = TensorFusionModel(["1", "3"])
            self.tf_23 = TensorFusionModel(["2", "3"])
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

        # ------------------------------------------------------------------
        # (D)   Triple interaction
        # ------------------------------------------------------------------
        if self.interaction_order == 3:
            self.tf_123 = TensorFusionModel(["1", "2", "3"])
            self.normalize_triple = Normalize3(
                feature_dim_dict=feature_dim_dict_triple,
                track_running_stats=track_running_stats,
                mod_index="123"
            )

        # ------------------------------------------------------------------
        # (E)   Build MKL‑Fusion input‑dim mapping
        # ------------------------------------------------------------------
        in_feats = {
            "z1": dim_dict_single["1"],
            "z2": dim_dict_single["2"],
            "z3": dim_dict_single["3"],
        }
        if self.interaction_order >= 2:
            in_feats |= {
                "z12": feature_dim_dict_triple["12"],
                "z13": feature_dim_dict_triple["13"],
                "z23": feature_dim_dict_triple["23"],
            }
        if self.interaction_order == 3:
            in_feats["z123"] = feature_dim_dict_triple["123"]

        # ------------------------------------------------------------------
        # (E)   Final fusion layer
        # ------------------------------------------------------------------
        self.intense_reg_rate = intense_reg_rate
        self.mkl_fusion = MKLFusion(
            in_features=in_feats,
            out_features=out_features,
            bias=True,
            reg_rate=intense_reg_rate,
            intense_p=intense_p
        )
    # ---------------------------------------------------------------------
    #                            forward pass
    # ---------------------------------------------------------------------
    def forward(self, z_dict: dict[str, Tensor]) -> torch.Tensor:
        """
        Args:
        z_dict (dict[str, torch.Tensor]): Dictionary with keys "1", "2", "3"
            representing individual modality embeddings.
        Returns:
            final_out: [B, out_features], e.g., classification logits or regression output.
        """

        # 1) BN over single‑modality embeddings --------------------------------
        z1 = self.bn_single["1"](z_dict["1"])
        z2 = self.bn_single["2"](z_dict["2"])
        z3 = self.bn_single["3"](z_dict["3"])

        tensor_list = [z1, z2, z3]

        # 1.1) optional projection -------------------------------------------
        if self.tf_latent_dim is not None:
            z1 = self.proj_single["1"](z1)
            z2 = self.proj_single["2"](z2)
            z3 = self.proj_single["3"](z3)

        z_dict_norm = {"1": z1, "2": z2, "3": z3}

        # 2) Pairwise interactions if specified (interaction_order >=2)
        if self.interaction_order >= 2:
            z12 = self.tf_12(z_dict_norm)
            z13 = self.tf_13(z_dict_norm)
            z23 = self.tf_23(z_dict_norm)

            z12 = self.bn_pair["12"](z12)
            z13 = self.bn_pair["13"](z13)
            z23 = self.bn_pair["23"](z23)

            tensor_list.extend([z12, z13, z23])

        # 3) Triple interaction if specified (interaction_order ==3)
        if self.interaction_order == 3:
            z123 = self.tf_123(z_dict_norm)
            # The triple normalization can use the already computed pairwise outputs
            input_dict = {"12": z12, "13": z13, "23": z23, "123": z123}
            pre_fusion_dict = {"1": z1, "2": z2, "3": z3}
            z123_norm = self.normalize_triple(input_dict, pre_fusion_dict)
            tensor_list.append(z123_norm)

        # 4) Fuse all selected representations
        final_out = self.mkl_fusion(tensor_list)
        return final_out

    # -------------------------- helpers --------------------------------------
    def get_regularizer(self) -> Tensor:
        return self.mkl_fusion.regularizer()

    def get_relevance_score(self) -> Dict[str, float]:
        return self.mkl_fusion.scores()
