# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, List, Sequence

import torch
import torch.nn as nn
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import gather, get_rank
import torch.nn.functional as F
from solo.losses.oursloss import ours_loss_func
from solo.utils.metrics import corrcoef, pearsonr_cor
import ipdb


class BarlowTwins(BaseMethod):
    def __init__(
        self, 
        proj_hidden_dim: int, 
        proj_output_dim: int, 
        lamb: float, scale_loss: float, 
        lam: float,
        tau_decor: float,
        our_loss: str,
        **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

        super().__init__(**kwargs)

        self.lam = lam
        self.tau_decor = tau_decor
        self.our_loss = our_loss

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwins, BarlowTwins).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)

        # our loss
        parser.add_argument("--lam", type=float, default=0.1)
        parser.add_argument("--tau_decor", type=float, default=0.1)
        parser.add_argument("--our_loss", type=str, default='True')
        
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X, *args, **kwargs):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        # ------- barlow twins loss -------
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        self.log("train_barlow_loss", barlow_loss, on_epoch=True, sync_dist=True)

        # ipdb.set_trace()

        ### add our loss
        original_loss = barlow_loss
        if self.our_loss=='True':
            our_loss = ours_loss_func(z1, z2, indexes=batch[0].repeat(self.num_large_crops + self.num_small_crops), tau_decor = self.tau_decor)
            total_loss = self.lam*our_loss + (1-self.lam)*original_loss
        elif self.our_loss=='False':
            total_loss = original_loss
        else:
            assert self.our_loss in ['True', 'False'], 'Input of our_loss is only True or False'
        ###

        with torch.no_grad():
            z_std = F.normalize(torch.stack((z1,z2)), dim=-1).std(dim=1).mean()
            corr_z = torch.abs(corrcoef(z1, z2).diag(-1)).mean()
            pear_z = pearsonr_cor(z1, z2).mean()
            corr_feats = torch.abs(corrcoef(feats1, feats2).diag(-1)).mean()
            pear_feats = pearsonr_cor(feats1, feats2).mean()

        ### new metrics
        metrics = {
            "Logits/avg_sum_logits_Z": (torch.stack((z1,z2))).sum(-1).mean(),
            "Logits/avg_sum_logits_Z_normalized": F.normalize(torch.stack((z1,z2)), dim=-1).sum(-1).mean(),

            "Logits/logits_Z_max": (torch.stack((z1,z2))).max(),
            "Logits/logits_Z_min": (torch.stack((z1,z2))).min(),

            "Logits/var_Z": (torch.stack((z1,z2))).var(-1).mean(),

            "Logits/logits_Z_normalized_max": F.normalize(torch.stack((z1,z2)), dim=-1).max(),
            "Logits/logits_Z_normalized_min": F.normalize(torch.stack((z1,z2)), dim=-1).min(),

            "MeanVector/mean_vector_Z_max": (torch.stack((z1,z2))).mean(1).max(),
            "MeanVector/mean_vector_Z_min": (torch.stack((z1,z2))).mean(1).min(),
            "MeanVector/mean_vector_Z_normalized_max": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_Z_normalized_min": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).min(),

            "MeanVector/norm_vector_Z": (torch.stack((z1,z2))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z_normalized": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).mean(0).norm(),

            "Backbone/var": (torch.stack((feats1,feats2))).var(-1).mean(),
            "Backbone/max": (torch.stack((feats1,feats2))).max(),

            "train_z_std": z_std,
            "Corr/corr_z": corr_z,
            "Corr/pear_z": pear_z,
            "Corr/corr_feats": corr_feats,
            "Corr/pear_feats": pear_feats,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        ### new metrics

        return total_loss + class_loss
