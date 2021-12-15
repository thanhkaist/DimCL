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
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather, get_rank
from solo.losses.oursloss import ours_loss_func


class BYOL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        lam: float,
        tau_decor: float,
        our_loss: str,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        self.lam = lam
        self.tau_decor = tau_decor
        self.our_loss = our_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BYOL, BYOL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("byol")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # our loss
        parser.add_argument("--lam", type=float, default=0.1)
        parser.add_argument("--tau_decor", type=float, default=0.1)
        parser.add_argument("--our_loss", type=str, default='True')


        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def _shared_step(
        self, batch: Sequence[Any], feats: List[torch.Tensor], momentum_feats: List[torch.Tensor]
    ) -> torch.Tensor:

        Z = [self.projector(f) for f in feats]
        P = [self.predictor(z) for z in Z]

        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        # for v1 in range(self.num_large_crops):
        #     for v2 in np.delete(range(self.num_crops), v1):
        #         neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        ### add our loss
        byol_loss = byol_loss_func(P[1], Z_momentum[0]) + byol_loss_func(P[0], Z_momentum[1])
        our_loss = ours_loss_func(Z[0], Z[1], indexes=batch[0].repeat(self.num_large_crops + self.num_small_crops), tau_decor = self.tau_decor)
        if self.our_loss=='True':
            neg_cos_sim = self.lam*our_loss + (1-self.lam)*byol_loss
            print('True sdfsf')
        elif self.our_loss=='False':
            neg_cos_sim = byol_loss
            print('False sdfsf')
        else:
            assert self.our_loss in ['True', 'False'], 'Input of our_loss is only True or False'
        ###

        

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
        
        ### new metrics
        metrics = {
            "Logits/avg_sum_logits_P": (torch.stack(P[: self.num_large_crops])).sum(-1).mean(),
            "Logits/avg_sum_logits_P_normalized": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).sum(-1).mean(),
            "Logits/avg_sum_logits_Z": (torch.stack(Z[: self.num_large_crops])).sum(-1).mean(),
            "Logits/avg_sum_logits_Z_normalized": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).sum(-1).mean(),

            "Logits/logits_P_max": (torch.stack(P[: self.num_large_crops])).max(),
            "Logits/logits_P_min": (torch.stack(P[: self.num_large_crops])).min(),
            "Logits/logits_Z_max": (torch.stack(Z[: self.num_large_crops])).max(),
            "Logits/logits_Z_min": (torch.stack(Z[: self.num_large_crops])).min(),

            "Logits/var_P": (torch.stack(P[: self.num_large_crops])).var(-1).mean(),
            "Logits/var_Z": (torch.stack(Z[: self.num_large_crops])).var(-1).mean(),

            "Logits/logits_P_normalized_max": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).max(),
            "Logits/logits_P_normalized_min": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).min(),
            "Logits/logits_Z_normalized_max": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).max(),
            "Logits/logits_Z_normalized_min": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).min(),

            "MeanVector/mean_vector_P_max": (torch.stack(P[: self.num_large_crops])).mean(1).max(),
            "MeanVector/mean_vector_P_min": (torch.stack(P[: self.num_large_crops])).mean(1).min(),
            "MeanVector/mean_vector_P_normalized_max": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_P_normalized_min": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).mean(1).min(),

            "MeanVector/mean_vector_Z_max": (torch.stack(Z[: self.num_large_crops])).mean(1).max(),
            "MeanVector/mean_vector_Z_min": (torch.stack(Z[: self.num_large_crops])).mean(1).min(),
            "MeanVector/mean_vector_Z_normalized_max": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_Z_normalized_min": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).mean(1).min(),

            "MeanVector/norm_vector_P": (torch.stack(P[: self.num_large_crops])).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_P_normalized": F.normalize(torch.stack(P[: self.num_large_crops]), dim=-1).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z": (torch.stack(Z[: self.num_large_crops])).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z_normalized": F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).mean(1).mean(0).norm(),

            "Backbone/var": (torch.stack(feats[: self.num_large_crops])).var(-1).mean(),
            "Backbone/max": (torch.stack(feats[: self.num_large_crops])).max(),
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        ### new metrics

        return neg_cos_sim, z_std

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        neg_cos_sim, z_std = self._shared_step(batch, out["feats"], out["momentum_feats"])

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
