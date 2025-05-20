# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def MPDIoU_loss(pred_boxes, gt_boxes):
    """
    计算 MPDIoU 损失函数。
    Args:
        pred_boxes (torch.Tensor): 预测框，格式为 [x1, y1, x2, y2]。
        gt_boxes (torch.Tensor): 真实框，格式为 [x1, y1, x2, y2]。
        image_size (tuple): 图像的宽度和高度 (w, h)，用于归一化距离。
    Returns:
        torch.Tensor: MPDIoU 损失值。
    归一化处理：
        在归一化坐标系中，boxes 的坐标范围是 (0, 1)。
        因为 boxes 已经归一化，所以 w 和 h 也应归一化为 1.0。
    距离计算：
        计算左上角和右下角点的欧氏距离平方时，直接使用归一化坐标。
        由于 w 和 h 已归一化为 1.0，因此分母 𝑤2+ℎ2=2 w2+h2=2。
    """
    # 归一化 w 和 h，使其与 boxes 的坐标格式一致
    # 由于 boxes 是归一化的，因此 w 和 h 也应归一化
    w = torch.tensor(1.0, device='cuda')  # 归一化后的宽度
    h = torch.tensor(1.0, device='cuda')  # 归一化后的高度

    # 计算 IoU
    iou, _ = box_iou(pred_boxes, gt_boxes)

    # 计算左上角点的距离
    d1_sq = (pred_boxes[:, 0] - gt_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - gt_boxes[:, 1]) ** 2

    # 计算右下角点的距离
    d2_sq = (pred_boxes[:, 2] - gt_boxes[:, 2]) ** 2 + (pred_boxes[:, 3] - gt_boxes[:, 3]) ** 2

    # 计算 MPDIoU
    mpdiou = iou - (d1_sq / (w ** 2 + h ** 2)) - (d2_sq / (w ** 2 + h ** 2))

    # 返回 MPDIoU 损失
    return 1 - mpdiou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
