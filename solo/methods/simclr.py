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
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import gather, get_rank
import torch.nn.functional as F
from solo.losses.oursloss import ours_loss_func
from solo.utils.metrics import corrcoef, pearsonr_cor
import ipdb

class SimCLR(BaseMethod):
    def __init__(
        self, 
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        lam: float,
        tau_decor: float,
        our_loss: str,
        **kwargs):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(**kwargs)

        self.lam = lam
        self.tau_decor = tau_decor
        self.our_loss = our_loss

        self.temperature = temperature

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR, SimCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # our loss
        parser.add_argument("--lam", type=float, default=0.1)
        parser.add_argument("--tau_decor", type=float, default=0.1)
        parser.add_argument("--our_loss", type=str, default='True')

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]

        # z = torch.cat([self.projector(f) for f in feats])
        Z = [self.projector(f) for f in feats]

        z = torch.cat((Z))

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        # ipdb.set_trace()

        ### add our loss
        original_loss = nce_loss
        if self.our_loss=='True':
            our_loss = ours_loss_func(Z[0], Z[1], indexes=batch[0].repeat(self.num_large_crops + self.num_small_crops), tau_decor = self.tau_decor)
            total_loss = self.lam*our_loss + (1-self.lam)*original_loss
        elif self.our_loss=='False':
            total_loss = original_loss
        else:
            assert self.our_loss in ['True', 'False'], 'Input of our_loss is only True or False'
        ###

        self.log("train_nce_loss", total_loss, on_epoch=True, sync_dist=True)
        

        ### new metrics
        z1, z2 = Z[0], Z[1]
        with torch.no_grad():
            z_std = F.normalize(torch.stack((z1,z2)), dim=-1).std(dim=1).mean()
        
        corr = torch.abs(corrcoef(Z[0], Z[1]).diag(-1)).mean()
        pear = pearsonr_cor(Z[0], Z[1]).mean()

        metrics = {
            "Logits/avg_sum_logits_Z": (torch.stack((z1,z2))).sum(-1).mean(),
            "Logits/avg_sum_logits_Z_normalized": F.normalize(torch.stack((z1,z2)), dim=-1).sum(-1).mean(),
            "Logits/logits_Z_max": (torch.stack((z1,z2))).max(),
            "Logits/logits_Z_min": (torch.stack((z1,z2))).min(),

            "Logits/logits_Z_normalized_max": F.normalize(torch.stack((z1,z2)), dim=-1).max(),
            "Logits/logits_Z_normalized_min": F.normalize(torch.stack((z1,z2)), dim=-1).min(),

            "MeanVector/mean_vector_Z_max": (torch.stack((z1,z2))).mean(1).max(),
            "MeanVector/mean_vector_Z_min": (torch.stack((z1,z2))).mean(1).min(),
            "MeanVector/mean_vector_Z_normalized_max": F.normalize(torch.stack((z1,z2))).mean(1).max(),
            "MeanVector/mean_vector_Z_normalized_min": F.normalize(torch.stack((z1,z2))).mean(1).min(),

            "MeanVector/norm_vector_Z": (torch.stack((z1,z2))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z_normalized": F.normalize(torch.stack((z1,z2))).mean(1).mean(0).norm(),

            "Backbone/var": (torch.stack(out["feats"])).var(-1).mean(),
            "Backbone/max": (torch.stack(out["feats"])).max(),
            "Logits/var_Z": (torch.stack((z1,z2))).var(-1).mean(),

            "train_z_std": z_std,
            "Corr/corr": corr,
            "Corr/pear": pear,

        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        ### new metrics

        return total_loss + class_loss
