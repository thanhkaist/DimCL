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
from solo.losses.moco import moco_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather, get_rank
from solo.losses.oursloss import ours_loss_func
from solo.utils.metrics import corrcoef, pearsonr_cor


class MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        lam: float,
        tau_decor: float,
        our_loss: str,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)

        self.lam = lam
        self.tau_decor = tau_decor
        self.our_loss = our_loss

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MoCoV2Plus, MoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        # our loss
        parser.add_argument("--lam", type=float, default=0.1)
        parser.add_argument("--tau_decor", type=float, default=0.1)
        parser.add_argument("--our_loss", type=str, default='True')
        

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        q1_ori = self.projector(feats1)
        q2_ori = self.projector(feats2)
        q1 = F.normalize(q1_ori, dim=-1)
        q2 = F.normalize(q2_ori, dim=-1)

        with torch.no_grad():
            k1_ori = self.momentum_projector(momentum_feats1)
            k2_ori = self.momentum_projector(momentum_feats2)
            k1 = F.normalize(k1_ori, dim=-1)
            k2 = F.normalize(k2_ori, dim=-1)

        # ------- contrastive loss -------
        # symmetric
        queue = self.queue.clone().detach()
        nce_loss = (
            moco_loss_func(q1, k2, queue[1], self.temperature)
            + moco_loss_func(q2, k1, queue[0], self.temperature)
        ) / 2

        ### add our loss
        original_loss = nce_loss
        if self.our_loss=='True':
            our_loss = ours_loss_func(q1_ori, q2_ori, indexes=batch[0].repeat(self.num_large_crops + self.num_small_crops), tau_decor = self.tau_decor)
            total_loss = self.lam*our_loss + (1-self.lam)*original_loss
        elif self.our_loss=='False':
            total_loss = original_loss
        else:
            assert self.our_loss in ['True', 'False'], 'Input of our_loss is only True or False'
        ###

        # ------- update queue -------
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        self.log("train_nce_loss", total_loss, on_epoch=True, sync_dist=True)

        with torch.no_grad():
            z_std = F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).std(dim=1).mean()
            corr_z = torch.abs(corrcoef(q1_ori, q2_ori).diag(-1)).mean()
            pear_z = pearsonr_cor(q1_ori, q2_ori).mean()
            corr_feats = torch.abs(corrcoef(feats1, feats2).diag(-1)).mean()
            pear_feats = pearsonr_cor(feats1, feats2).mean()

        ### new metrics
        metrics = {
            "Logits/avg_sum_logits_P": (torch.stack((q1_ori,q2_ori))).sum(-1).mean(),
            "Logits/avg_sum_logits_P_normalized": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).sum(-1).mean(),
            "Logits/avg_sum_logits_Z": (torch.stack((k1_ori,k2_ori))).sum(-1).mean(),
            "Logits/avg_sum_logits_Z_normalized": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).sum(-1).mean(),
            "Logits/logits_P_max": (torch.stack((q1_ori,q2_ori))).max(),
            "Logits/logits_P_min": (torch.stack((q1_ori,q2_ori))).min(),
            "Logits/logits_Z_max": (torch.stack((k1_ori,k2_ori))).max(),
            "Logits/logits_Z_min": (torch.stack((k1_ori,k2_ori))).min(),

            "Logits/logits_P_normalized_max": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).max(),
            "Logits/logits_P_normalized_min": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).min(),
            "Logits/logits_Z_normalized_max": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).max(),
            "Logits/logits_Z_normalized_min": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).min(),

            "MeanVector/mean_vector_P_max": (torch.stack((q1_ori,q2_ori))).mean(1).max(),
            "MeanVector/mean_vector_P_min": (torch.stack((q1_ori,q2_ori))).mean(1).min(),
            "MeanVector/mean_vector_P_normalized_max": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_P_normalized_min": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).mean(1).min(),

            "MeanVector/mean_vector_Z_max": (torch.stack((k1_ori,k2_ori))).mean(1).max(),
            "MeanVector/mean_vector_Z_min": (torch.stack((k1_ori,k2_ori))).mean(1).min(),
            "MeanVector/mean_vector_Z_normalized_max": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_Z_normalized_min": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).mean(1).min(),

            "MeanVector/norm_vector_P": (torch.stack((q1_ori,q2_ori))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_P_normalized": F.normalize(torch.stack((q1_ori,q2_ori)), dim=-1).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z": (torch.stack((k1_ori,k2_ori))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z_normalized": F.normalize(torch.stack((k1_ori,k2_ori)), dim=-1).mean(1).mean(0).norm(),

            "Logits/var_P": (torch.stack((q1_ori,q2_ori))).var(-1).mean(),
            "Logits/var_Z": (torch.stack((q1_ori,q2_ori))).var(-1).mean(),

            "Backbone/var": (torch.stack((feats1, feats2))).var(-1).mean(),
            "Backbone/max": (torch.stack((feats1, feats2))).max(),

            "train_z_std": z_std,
            "Corr/corr_z": corr_z,
            "Corr/pear_z": pear_z,
            "Corr/corr_feats": corr_feats,
            "Corr/pear_feats": pear_feats,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        ### new metrics

        return total_loss + class_loss
