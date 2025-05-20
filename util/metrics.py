import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
from util import box_ops


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir="result", names=(), eps=1e-16, prefix=""):
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # 确保 unique_classes 和每个类别都有足够的数据
    if len(unique_classes) == 0:
        print("没有找到任何类别。")
        return

    # Create Precision - Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    if len(tp.shape) == 1:
        ap, p, r = np.zeros((nc,)), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    else:
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        # recall = tpc / (n_l + eps)  # recall curve
        # r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        recall = tpc / (n_l + eps)
        if len(recall.shape) == 1:
            r[ci] = np.interp(-px, -conf[i], recall, left=0)
        else:
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        if len(precision.shape) == 1:
            p[ci] = np.interp(-px, -conf[i], precision, left=1)
        else:
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP from recall - precision curve
        ap[ci], mpre, mrec = compute_ap(recall, precision)

        # 修改为不论类别数量，都将数据添加到 py 中
        if plot:
            py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict

    # 绘制 PR 曲线并确保保存路径存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        if not py:
            print("警告: `py` 为空，无法绘制 PR 曲线。")
        else:
            # 这里假设存在绘图函数plot_pr_curve、plot_mc_curve，实际使用时需要实现这些函数
            plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
            plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
            plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
            plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def detr_ap_per_class(detr_outputs_list, targets_list, num_classes, plot=False, save_dir="result", names=(), eps=1e-16, prefix=""):
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []

    for detr_outputs, targets in zip(detr_outputs_list, targets_list):
        batch_size = detr_outputs["pred_boxes"].shape[0]
        num_queries = detr_outputs["pred_boxes"].shape[1]

        out_logits = detr_outputs["pred_logits"]
        out_bbox = detr_outputs['pred_boxes']

        '''
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # 将模型预测的目标框坐标从相对坐标（x0,y0,h,w)->绝对坐标(x0,y0,x1,y1)
        out_bbox = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        out_bbox = out_bbox * scale_fct[:, None, :]
        '''

        for b in range(batch_size):
            confidences = torch.softmax(out_logits[b], dim=-1)
            pred_classes = torch.argmax(confidences, dim=-1)
            pred_boxes = out_bbox[b]
            target_boxes_b = targets[b]["boxes"]
            target_classes_b = targets[b]["labels"]

            tp = []
            conf = []
            pred_cls = []
            target_cls = []

            # print(f'target boxes {target_boxes_b}')
            # print(f'pred boxes {pred_boxes}')
            target_boxes_b = box_ops.box_cxcywh_to_xyxy(target_boxes_b)
            pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            iou_matrix = box_iou(target_boxes_b, pred_boxes)
            # print('***********************iou_matrix***************************************')
            # print(iou_matrix)

            for i in range(num_queries):
                max_iou_idx = torch.argmax(iou_matrix[:, i])
                if iou_matrix[max_iou_idx, i] > 0.5 and pred_classes[i] == target_classes_b[max_iou_idx]:
                    tp.append(1)
                else:
                    tp.append(0)
                conf.append(confidences[i, pred_classes[i]].item())
                pred_cls.append(pred_classes[i].item())
                target_cls.append(target_classes_b[max_iou_idx].item())

            tp_all.extend(tp)
            conf_all.extend(conf)
            pred_cls_all.extend(pred_cls)
            target_cls_all.extend(target_cls)

    tp_all = np.array(tp_all)
    conf_all = np.array(conf_all)
    pred_cls_all = np.array(pred_cls_all)
    target_cls_all = np.array(target_cls_all)

    return ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all, plot=plot, save_dir=save_dir, names=names, eps=eps, prefix=prefix)


'''
def compute_pr_from_detr_outputs(detr_outputs_list, targets_list, num_classes, plot=False, save_dir="result", names=(), eps=1e-16, prefix=""):
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []

    for detr_outputs, targets in zip(detr_outputs_list, targets_list):
        pred_logits = detr_outputs["pred_logits"]
        pred_boxes = detr_outputs["pred_boxes"]

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 将模型预测的目标框坐标从相对坐标（x0,y0,h,w)->绝对坐标(x0,y0,x1,y1)
        pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = pred_boxes * scale_fct[:, None, :]

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
                if iou_matrix[max_iou_idx, i] > 0.5 and pred_classes[b, i] == target_classes[max_iou_idx]:
                    tp_all.append(1)  # 真阳性
                else:
                    tp_all.append(0)  # 假阳性

                conf_all.append(confidences[b, i, pred_classes[b, i]].item())
                target_cls_all.append(target_classes[max_iou_idx].item() if max_iou_idx < len(target_classes) else -1)
                pred_cls_all.append(pred_classes[b, i].item())

    tp_all = np.array(tp_all)
    conf_all = np.array(conf_all)

    # 计算精确度和召回率
    return ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all, plot, save_dir, names, prefix)  # 根据需要更改pos_label
'''


def rescale_bboxes(out_bbox, size):
    # 把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def compute_ap(recall, precision):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[: - 1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre[1:], mrec[1:]


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
    # return np.conv2d(yp.reshape(1, -1), np.ones((1, nf)) / nf, mode="valid")[0]


# '''
def box_iou(boxes1, boxes2):
    """
    Return intersection - over - union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
# '''

'''
def box_iou(target_boxes, pred_boxes):
    target_boxes = target_boxes.unsqueeze(1)  # [num_targets, 1, 4]
    pred_boxes = pred_boxes.unsqueeze(0)  # [1, num_preds, 4]

    inter_x0 = torch.max(target_boxes[..., 0], pred_boxes[..., 0])
    inter_y0 = torch.max(target_boxes[..., 1], pred_boxes[..., 1])
    inter_x1 = torch.min(target_boxes[..., 2], pred_boxes[..., 2])
    inter_y1 = torch.min(target_boxes[..., 3], pred_boxes[..., 3])

    inter_area = torch.clamp(inter_x1 - inter_x0, min=0) * torch.clamp(inter_y1 - inter_y0, min=0)
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])

    union_area = target_area + pred_area - inter_area
    iou = inter_area / (union_area + 1e-16)

    return iou  # 返回 [num_targets, num_preds]
'''

def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """Plots precision-recall curve, optionally per class, saving to `save_dir`; `px`, `py` are lists, `ap` is Nx2
    array, `names` optional.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 32:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            # ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i]:.3f}")  # 将 ap[i, 0] 改为 ap[i]
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap.mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 32:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
