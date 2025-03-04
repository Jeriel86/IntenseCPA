import torch
import torch.nn as nn
from typing import Optional


class VectorWiseBatchNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        factory_kwargs = {"device": device, "dtype": dtype}
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer("running_var", torch.ones(1, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            self.running_var: Optional[torch.Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            n = input.size(0)
            mean = torch.mean(input, dim=0)
            # use biased var in train
            var = torch.sum((input - mean) ** 2) / n
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )
                # update running_var with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean) / (torch.sqrt(var + self.eps))

        return input

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 2:
            raise ValueError(f"expected 2D input (got {input.dim()}D input)")


class Normalize3(nn.Module):
    def __init__(
        self,
        feature_dim_dict: dict[str, int],
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        mod_index: str = "123",
    ):
        super().__init__()
        self.feature_dim_dict = feature_dim_dict
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        factory_kwargs = {"device": device, "dtype": dtype}
        if self.track_running_stats:
            self.register_buffer(
                "running_mean",
                torch.zeros(feature_dim_dict[mod_index], **factory_kwargs),
            )
            self.register_buffer(
                "running_mean_12", torch.zeros(feature_dim_dict["12"], **factory_kwargs)
            )
            self.register_buffer(
                "running_mean_13", torch.zeros(feature_dim_dict["13"], **factory_kwargs)
            )
            self.register_buffer(
                "running_mean_23", torch.zeros(feature_dim_dict["23"], **factory_kwargs)
            )
            self.register_buffer("running_var", torch.ones(1, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            self.running_mean_12: Optional[torch.Tensor]
            self.running_mean_13: Optional[torch.Tensor]
            self.running_mean_23: Optional[torch.Tensor]
            self.running_var: Optional[torch.Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_mean_12", None)
            self.register_buffer("running_mean_13", None)
            self.register_buffer("running_mean_23", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()
        self.mod_index = mod_index

    def forward(
        self, input: dict[str, torch.Tensor], pre_fusion_tens: dict[str, torch.Tensor]
    ):
        inp = input[self.mod_index]
        self._check_input_dim(inp)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean_12 = torch.mean(input["12"], dim=0)
            mean_13 = torch.mean(input["13"], dim=0)
            mean_23 = torch.mean(input["23"], dim=0)
            n = inp.size(0)
            mean_123 = torch.mean(inp, dim=0)
            # use biased var in train
            mean = self.compute_mean(
                pre_fusion_tens,
                mean_12=mean_12,
                mean_13=mean_13,
                mean_23=mean_23,
                mean_123=mean_123,
            )
            var = torch.sum((inp - mean) ** 2) / n
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean_123
                    + (1 - exponential_average_factor) * self.running_mean
                )
                self.running_mean_12 = (
                    exponential_average_factor * mean_12
                    + (1 - exponential_average_factor) * self.running_mean_12
                )
                self.running_mean_13 = (
                    exponential_average_factor * mean_13
                    + (1 - exponential_average_factor) * self.running_mean_13
                )
                self.running_mean_23 = (
                    exponential_average_factor * mean_23
                    + (1 - exponential_average_factor) * self.running_mean_23
                )
                # update running_var with unbiased var
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )
        else:
            mean_123 = self.running_mean
            mean_12 = self.running_mean_12
            mean_13 = self.running_mean_13
            mean_23 = self.running_mean_23
            var = self.running_var
            mean = self.compute_mean(
                pre_fusion_tens,
                mean_12=mean_12,
                mean_13=mean_13,
                mean_23=mean_23,
                mean_123=mean_123,
            )
        inp = (inp - mean) / (torch.sqrt(var + self.eps))
        return inp

    def compute_mean(
        self,
        pre_fusion_tens: dict[str, torch.Tensor],
        mean_12,
        mean_13,
        mean_23,
        mean_123,
    ):
        out_12_3 = self.compute_tensor_product([mean_12, pre_fusion_tens["3"]])
        out_12_3 = torch.flatten(out_12_3, start_dim=1)
        out_13_2 = self.compute_tensor_product([mean_13, pre_fusion_tens["2"]])
        out_13_2 = torch.flatten(out_13_2, start_dim=1)
        out_23_1 = self.compute_tensor_product([mean_23, pre_fusion_tens["1"]])
        out_23_1 = torch.flatten(out_23_1, start_dim=1)
        mean = mean_123 + out_12_3 + out_13_2 + out_23_1
        return mean

    def compute_tensor_product(self, inp: list[torch.Tensor]) -> torch.Tensor:
        if len(inp) == 2:
            return torch.einsum("...i,bj->bij", inp)
        else:
            raise ValueError("Tensor product is only supported for 2 batch vectors.")

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_mean_12.zero_()
            self.running_mean_13.zero_()
            self.running_mean_23.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 2:
            raise ValueError(f"expected 2D input (got {input.dim()}D input)")
