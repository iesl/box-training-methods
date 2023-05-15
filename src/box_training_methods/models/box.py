from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from .temps import convert_float_to_const_temp
from box_training_methods.utils import tiny_value_of_dtype
from box_training_methods import metric_logger

__all__ = [
    "BoxMinDeltaSoftplus",
    "TBox",
]


eps = tiny_value_of_dtype(torch.float)


# TODO rename to BoxCenterDeltaSoftplus
class BoxMinDeltaSoftplus(Module):
    def __init__(self, num_entity, dim, volume_temp=1.0, intersection_temp=1.0):
        super().__init__()
        self.centers = torch.nn.Embedding(num_entity, dim)
        self.sidelengths = torch.nn.Embedding(num_entity, dim)
        self.centers.weight.data.uniform_(-0.1, 0.1)
        self.sidelengths.weight.data.zero_()

        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060

    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.softplus(Z - z - self.softplus_const)), dim=-1,
        )
        return log_vol

    def embedding_lookup(self, idx):
        center = self.centers(idx)
        length = self.softplus(self.sidelengths(idx))
        z = center - length
        Z = center + length
        return z, Z

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idxs):
        """
        :param idxs: Tensor of shape (..., 2) (N, K+1, 2) during training or (N, 2) during testing
        :return: log prob of shape (..., )
        """
        e1_min, e1_max = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        log_overlap_volume = self.log_volume(meet_min, meet_max)
        log_rhs_volume = self.log_volume(e2_min, e2_max)

        return log_overlap_volume - log_rhs_volume

    def forward_log_overlap_volume(self, idxs):
        """
        :param idxs: Tensor of shape (N, 2)
        :return: log of overlap volume, shape (N, )
        """
        e1_min, e1_max = self.embedding_lookup(idxs[..., 0])
        e2_min, e2_max = self.embedding_lookup(idxs[..., 1])

        meet_min, meet_max = self.gumbel_intersection(e1_min, e1_max, e2_min, e2_max)

        log_overlap_volume = self.log_volume(meet_min, meet_max)

        return log_overlap_volume

    def forward_log_marginal_volume(self, idxs):
        """
        :param idxs: Tensor of shape (N, )
        :return: log of marginal volume, shape (N, )
        """
        e_min, e_max = self.embedding_lookup(idxs)
        log_volume = self.log_volume(e_min, e_max)

        return log_volume


class TBox(Module):
    """
    Box embedding model where the temperatures can (optionally) be trained.

    In this model, the self.boxes parameter is of shape (num_entity, 2, dim), where self.boxes[i,:,k] are location
    parameters for Gumbel distributions representing the corners of the ith box in the kth dimension.
        self.boxes[i,0,k] is the location parameter mu_z for a MaxGumbel distribution
        self.boxes[i,1,k] represents -mu_Z, i.e. negation of location parameter, for a MinGumbel distribution
    This rather odd convention is chosen to maximize speed / ease of computation.

    Note that with this parameterization, we allow the location parameter to "flip around", i.e. mu_z > mu_Z.
    This is completely reasonable, from the GumbelBox perspective (in fact, a bit more reasonable than requiring
    mu_Z > mu_z, as this means the distributions are no longer independent).

    :param num_entities: Number of entities to create box embeddings for (eg. number of nodes).
    :param dim: Embedding dimension (i.e. boxes will be in RR^dim).
    :param intersection_temp: Temperature for intersection LogSumExp calculations
    :param volume_temp: Temperature for volume LogSumExp calculations
        Note: Temperatures can either be either a float representing a constant (global) temperature,
        or a Module which, when called, takes a LongTensor of indices and returns their temps.
    """

    def __init__(
        self,
        num_entities: int,
        dim: int,
        intersection_temp: Union[Module, float] = 0.01,
        volume_temp: Union[Module, float] = 1.0,
        hard_box: bool = False,
    ):
        super().__init__()
        self.boxes = Parameter(
            torch.sort(torch.randn((num_entities, 2, dim)), dim=-2).values
            * torch.tensor([1, -1])[None, :, None]
        )
        self.intersection_temp = convert_float_to_const_temp(intersection_temp)
        self.volume_temp = convert_float_to_const_temp(volume_temp)
        self.hard_box = hard_box

    def forward(
        self, idxs: LongTensor, instances: Optional[LongTensor] = None
    ) -> Union[Tuple[Tensor, Dict[str, Tensor]], Tensor]:
        """
        A version of the forward pass that is slightly more performant.
        :param idxs: Tensor of shape (..., 2) indicating edges, i.e. [...,0] -> [..., 1] is an edge
        :param instances: instance boxes of shape (..., 2 (min/-max), dim)
        :returns: FloatTensor representing the energy of the edges in `idxs`
        """
        boxes = self.boxes[idxs]  # shape (..., 2, 2 (min/-max), dim) or (..., 2 (min/-max), dim) if instances are provided

        if self.hard_box and self.training:
            return boxes
        if instances is not None:
            boxes = torch.stack([boxes, instances], dim=-3)  # labels -> instances

        intersection_temp = self.intersection_temp(idxs).mean(dim=-3, keepdim=True)
        volume_temp = self.volume_temp(idxs).mean(dim=-3, keepdim=False)

        # calculate Gumbel intersection
        intersection = intersection_temp * torch.logsumexp(
            boxes / intersection_temp, dim=-3, keepdim=True
        )
        intersection = torch.max(
            torch.cat((intersection, boxes), dim=-3), dim=-3
        ).values
        # combine intersections and marginals, since we are going to perform the same operations on both
        intersection_and_marginal = torch.stack(
            (intersection, boxes[..., 1, :, :]), dim=-3
        )
        # calculating log volumes
        # keep in mind that the [...,1,:] represents negative max, thus we negate it
        log_volumes = torch.sum(
            torch.log(
                volume_temp
                * F.softplus((-intersection_and_marginal.sum(dim=-2)) / volume_temp)
                + 1e-23
            ),
            dim=-1,
        )
        out = log_volumes[..., 0] - log_volumes[..., 1]

        if self.training and isinstance(metric_logger.metric_logger, WandBLogger):
            regularizer_terms = {
                "intersection_temp": self.intersection_temp(idxs).squeeze(-2),
                "volume_temp": self.volume_temp(idxs).squeeze(-2),
                "log_marginal_vol": log_volumes[..., 1],
                # "marginal_vol": log_volumes[..., 1].exp(),
                "side_length": -boxes.sum(dim=-2),
            }
            metrics_to_collect = {
                "pos": wandb.Histogram(out[..., 0].detach().exp().cpu()),
                "neg": wandb.Histogram(out[..., 1:].detach().exp().cpu()),
            }
            for k, v in regularizer_terms.items():
                if k == "intersection_temp":
                    metrics_to_collect[k] = v
                else:
                    metrics_to_collect[k] = wandb.Histogram(v.detach().cpu())

            metric_logger.metric_logger.collect(
                {f"[Train] {k}": v for k, v in metrics_to_collect.items()},
                overwrite=True,
            )
        return out
    
    def scores(self, instance_boxes, label_boxes, intersection_temp, volume_temp):
        """
        instance_boxes: (batch_size, 2 (min/-max), dim)
        label_boxes:    (batch_size, num_labels, 2 (min/-max), dim)
        """

        label_boxes = label_boxes.unsqueeze(dim=2)                                                                  # (batch_size, num_labels,       1, 2 (min/-max), dim)
        instance_boxes = torch.broadcast_to(instance_boxes.unsqueeze(dim=1).unsqueeze(dim=1), label_boxes.shape)    # (batch_size, 1 -> num_labels,  1, 2 (min/-max), dim)
        
        boxes = torch.cat([label_boxes, instance_boxes], dim=2)   # (batch_size, num_labels, 2 (u/v), 2 (min/-max), dim)

        # calculate Gumbel intersection
        intersection = intersection_temp * torch.logsumexp(
            boxes / intersection_temp, dim=-3, keepdim=True
        )
        intersection = torch.max(
            torch.cat((intersection, boxes), dim=-3), dim=-3
        ).values
        # combine intersections and marginals, since we are going to perform the same operations on both
        intersection_and_marginal = torch.stack(
            (intersection, boxes[..., 1, :, :]), dim=-3
        )
        # calculating log volumes
        # keep in mind that the [...,1,:] represents negative max, thus we negate it
        log_volumes = torch.sum(
            torch.log(
                volume_temp
                * F.softplus((-intersection_and_marginal.sum(dim=-2)) / volume_temp)
                + 1e-23
            ),
            dim=-1,
        )
        out = log_volumes[..., 0] - log_volumes[..., 1]
        return out
