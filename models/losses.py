import torch
import torch.nn.functional as F

@torch.jit.script
def batch_masked_sigmoid_focal_loss(inputs, targets, batch_out_mask, batch_tgt_mask, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    out_mask_float = batch_out_mask.to(focal_pos.dtype).detach()
    focal_pos, focal_neg = focal_pos * out_mask_float, focal_neg * out_mask_float

    tgt_mask_float = batch_tgt_mask.to(focal_pos.dtype).detach()

    loss = (torch.einsum("nc,mc->nm", focal_pos, targets * tgt_mask_float) + 
            torch.einsum("nc,mc->nm", focal_neg, (1 - targets) * tgt_mask_float))

    ## we only div the first dim
    return loss / batch_out_mask.sum(-1).unsqueeze(-1)

@torch.jit.script
def batch_masked_sigmoid_ce_loss(inputs, targets, batch_out_mask, batch_tgt_mask):
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
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    out_mask_float = batch_out_mask.to(pos.dtype).detach()
    pos, neg = pos * out_mask_float, neg * out_mask_float

    tgt_mask_float = batch_tgt_mask.to(pos.dtype).detach()

    loss = (torch.einsum("nc,mc->nm", pos, targets * tgt_mask_float) + 
            torch.einsum("nc,mc->nm", neg, (1 - targets) * tgt_mask_float))

    ## we only div the first dim
    return loss / batch_out_mask.sum(-1).unsqueeze(-1)

@torch.jit.script
def batch_masked_dice_loss(inputs, targets, batch_out_mask, batch_tgt_mask):
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

    out_mask_float = batch_out_mask.to(inputs.dtype).detach()
    inputs = inputs * out_mask_float
    
    tgt_mask_float = batch_tgt_mask.to(inputs.dtype).detach()
    targets = targets * tgt_mask_float
    
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

@torch.jit.script
def masked_sigmoid_focal_loss(inputs, targets, num_masks, loss_mask, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    mask_float = loss_mask.to(loss.dtype).detach()
    loss = mask_float * loss

    return loss.mean(1).sum() / num_masks

@torch.jit.script
def masked_sigmoid_ce_loss(inputs, targets, num_masks, loss_mask):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    mask_float = loss_mask.to(loss.dtype).detach()
    loss = mask_float * loss

    return (loss.sum(1) / mask_float.sum(1)).sum() / num_masks


@torch.jit.script
def masked_dice_loss(inputs, targets, num_masks, loss_mask):
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

    mask_float = loss_mask.to(inputs.dtype).detach()
    inputs = inputs * mask_float
    targets = targets * mask_float

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks


@torch.jit.script
def batch_masked_sigmoid_focal_fuzzy_loss(
        inputs, targets, 
        batch_out_mask, batch_tgt_mask, batch_tgt_seg, 
        scale_range: float = 0.8, alpha: float = 0.25, gamma: float = 2
    ):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    assert scale_range <= 1.0

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    out_mask_float = batch_out_mask.to(focal_pos.dtype).detach()
    focal_pos, focal_neg = focal_pos * out_mask_float, focal_neg * out_mask_float

    tgt_mask_float = batch_tgt_mask.to(focal_pos.dtype).detach()
    ## target weight
    target_center = (batch_tgt_seg[:, 1] - 1 + batch_tgt_seg[:, 0]).float() / 2
    target_index = torch.range(0, batch_tgt_mask.shape[-1] - 1, device=target_center.device).unsqueeze(0).repeat(batch_tgt_mask.shape[0], 1)
    
    ## pos
    abs_pos_range = torch.abs(target_index - target_center[:, None]) < ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / 2 * scale_range)
    all_range = (torch.abs(target_index - target_center[:, None]) < ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / 2 / scale_range)) & batch_tgt_mask
    fuzzy_range = (all_range ^ abs_pos_range) & batch_tgt_mask # xor
    
    pos_weight = torch.cos(torch.pi / ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / scale_range) * (target_index - target_center[:, None]))
    pos_weight = (pos_weight * (pos_weight > 0)) ** 0.5
    targets_pos = pos_weight * fuzzy_range +  targets * abs_pos_range

    loss = (torch.einsum("nc,mc->nm", focal_pos, targets_pos * tgt_mask_float) + 
            torch.einsum("nc,mc->nm", focal_neg, (1 - targets_pos) * tgt_mask_float))

    ## we only div the first dim
    return loss / batch_out_mask.sum(-1).unsqueeze(-1)


@torch.jit.script
def batch_masked_dice_fuzzy_loss(inputs, targets, batch_out_mask, batch_tgt_mask, batch_tgt_seg, scale_range: float = 0.8):
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

    out_mask_float = batch_out_mask.to(inputs.dtype).detach()
    inputs = inputs * out_mask_float
    
    tgt_mask_float = batch_tgt_mask.to(inputs.dtype).detach()

    target_center = (batch_tgt_seg[:, 1] - 1 + batch_tgt_seg[:, 0]).float() / 2
    target_index = torch.range(0, batch_tgt_mask.shape[-1] - 1, device=target_center.device).unsqueeze(0).repeat(batch_tgt_mask.shape[0], 1)

    abs_pos_range = torch.abs(target_index - target_center[:, None]) < ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / 2 * scale_range)
    all_range = (torch.abs(target_index - target_center[:, None]) < ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / 2 / scale_range)) & batch_tgt_mask
    fuzzy_range = (all_range ^ abs_pos_range) & batch_tgt_mask # xor

    pos_weight = torch.cos(torch.pi / ((batch_tgt_seg[:, 1] - batch_tgt_seg[:, 0])[:, None] / scale_range) * (target_index - target_center[:, None]))
    pos_weight = (pos_weight * (pos_weight > 0)) ** 0.5

    targets_pos = pos_weight * fuzzy_range +  targets * abs_pos_range
    targets_pos = targets_pos * tgt_mask_float
    
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets_pos)
    denominator = inputs.sum(-1)[:, None] + targets_pos.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


@torch.jit.script
def masked_sigmoid_focal_fuzzy_loss(inputs, targets, num_masks, loss_mask, tgt_segs, scale_range: float = 0.8, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()

    ## target weight
    target_center = (tgt_segs[:, 1] - 1 + tgt_segs[:, 0]).float() / 2
    target_index = torch.range(0, loss_mask.shape[-1] - 1, device=target_center.device).unsqueeze(0).repeat(loss_mask.shape[0], 1)
    
    ## pos
    abs_pos_range = torch.abs(target_index - target_center[:, None]) < ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / 2 * scale_range)
    all_range = (torch.abs(target_index - target_center[:, None]) < ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / 2 / scale_range)) & loss_mask
    fuzzy_range = (all_range ^ abs_pos_range) & loss_mask # xor
    
    pos_weight = torch.cos(torch.pi / ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / scale_range) * (target_index - target_center[:, None]))
    pos_weight = (pos_weight * (pos_weight > 0)) ** 0.5

    targets_pos = pos_weight * fuzzy_range +  targets * abs_pos_range

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_pos * loss_mask, reduction="none")

    p_t = prob * targets_pos + (1 - prob) * (1 - targets_pos)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets_pos + (1 - alpha) * (1 - targets_pos)
        loss = alpha_t * loss

    mask_float = loss_mask.to(loss.dtype).detach()
    loss = mask_float * loss

    return loss.mean(1).sum() / num_masks


@torch.jit.script
def masked_dice_fuzzy_loss(inputs, targets, num_masks, loss_mask, tgt_segs, scale_range: float = 0.8):
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

    mask_float = loss_mask.to(inputs.dtype).detach()
    inputs = inputs * mask_float

    ## target weight
    target_center = (tgt_segs[:, 1] - 1 + tgt_segs[:, 0]).float() / 2
    target_index = torch.range(0, loss_mask.shape[-1] - 1, device=target_center.device).unsqueeze(0).repeat(loss_mask.shape[0], 1)
    
    ## pos
    abs_pos_range = torch.abs(target_index - target_center[:, None]) < ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / 2 * scale_range)
    all_range = (torch.abs(target_index - target_center[:, None]) < ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / 2 / scale_range)) & loss_mask
    fuzzy_range = (all_range ^ abs_pos_range) & loss_mask # xor
    
    pos_weight = torch.cos(torch.pi / ((tgt_segs[:, 1] - tgt_segs[:, 0])[:, None] / scale_range) * (target_index - target_center[:, None]))
    pos_weight = (pos_weight * (pos_weight > 0)) ** 0.5

    targets_pos = pos_weight * fuzzy_range +  targets * abs_pos_range
    targets_pos = targets_pos * mask_float

    numerator = 2 * (inputs * targets_pos).sum(-1)
    denominator = inputs.sum(-1) + targets_pos.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks
