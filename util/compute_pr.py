import numpy as np
import torch


def compute_pr_from_detr_outputs(detr_outputs_list, targets_list, iou_threshold=0.5):
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []

    for detr_outputs, targets in zip(detr_outputs_list, targets_list):
        pred_logits = detr_outputs["pred_logits"]
        pred_boxes = detr_outputs["pred_boxes"]

        # 计算置信度
        confidences = torch.softmax(pred_logits, dim=-1)
        pred_classes = torch.argmax(confidences, dim=-1)

        batch_size = pred_boxes.shape[0]

        for b in range(batch_size):  # 遍历每个样本
            target_boxes = targets[b]["boxes"]
            target_classes = targets[b]["labels"]

            # 计算IoU
            iou_matrix = box_iou(target_boxes, pred_boxes[b])

            for i in range(pred_boxes[b].shape[0]):  # 对每个预测框
                if i >= len(target_boxes):  # 处理没有目标的情况
                    tp_all.append(0)
                    conf_all.append(confidences[b, i, pred_classes[b, i]].item())
                    pred_cls_all.append(pred_classes[b, i].item())
                    target_cls_all.append(-1)  # 无目标
                    continue

                max_iou_idx = torch.argmax(iou_matrix[:, i])
                if iou_matrix[max_iou_idx, i] > iou_threshold and pred_classes[b, i] == target_classes[max_iou_idx]:
                    tp_all.append(1)  # 真阳性
                else:
                    tp_all.append(0)  # 假阳性

                conf_all.append(confidences[b, i, pred_classes[b, i]].item())
                target_cls_all.append(target_classes[max_iou_idx].item() if max_iou_idx < len(target_classes) else -1)
                pred_cls_all.append(pred_classes[b, i].item())

    tp_all = np.array(tp_all)
    conf_all = np.array(conf_all)

    # 计算精确度和召回率
    precision, recall, _ = precision_recall_curve(target_cls_all, conf_all, pos_label=1)  # 根据需要更改pos_label

    return precision, recall