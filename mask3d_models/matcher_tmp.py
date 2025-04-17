# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

# from detectron2.projects.point_rend.point_features import point_sample

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    # pos = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction="none")
    # neg = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)  # type: torch.jit.ScriptModule

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, valid_bs):
        """More memory-friendly matching"""
        num_queries = outputs['pred_masks'][0].shape[-1]
        indices = []
        if valid_bs is None:
            valid_bs = torch.arange(len(targets))
        # Iterate through batch size
        for bs_index, bs in enumerate(valid_bs):

            out_prob = outputs["pred_logits"][bs].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = torch.tensor([0]*targets[bs_index].shape[-1]).long()### only one class, so 0 is chair, 1 is bg
            cost_class = -out_prob[:, tgt_ids].cpu()

            out_mask = outputs['pred_masks'][bs].T # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            # tgt_mask = targets[b]['masks'].to(out_mask)#[20, N]
            tgt_mask = targets[bs_index].to(out_mask).T#[20, N]

            point_idx = torch.arange(tgt_mask.shape[1], device=tgt_mask.device)

            with autocast(enabled=False):
                b = out_mask.float().cpu()
                out_mask = out_mask.float().cpu()#[50, sp_num]
                tgt_mask = tgt_mask.float().cpu()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask[:, point_idx], tgt_mask[:, point_idx])

            # Final cost matrix
            C = (2 * cost_class +5 * cost_mask + 2 * cost_dice)
            # C = (5 * cost_mask + 2 * cost_dice)
            C = C.reshape(num_queries, -1)#.cpu()
            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets, valid_bs=None):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, valid_bs)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = ["cost_class: {}".format(self.cost_class), "cost_mask: {}".format(self.cost_mask), "cost_dice: {}".format(self.cost_dice)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
