import matplotlib.pyplot as plt
import colorsys
import numpy as np
from typing import List

import posixpath
import pycocotools.mask as mask_util
from matplotlib import patches
from skimage.measure import find_contours
from matplotlib.patches import Polygon
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    #c==3 是因为颜色是三通道RGB
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    import matplotlib.pyplot as plt

    # plt.imshow(image.astype(np.uint8))
    # plt.show()
    return image
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(512, 512), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None,show_captions=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # if not N:
    #     print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax: #没有绘图的就用默认的
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        if show_bbox:#绘制矩形框
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=0.7, linestyle="solid",
                                edgecolor='red', facecolor='none')
            ax.add_patch(p)

        # Label
        if show_captions:
            if not captions:  # captions如果有值，表示框上面的字。没有就是label+score
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score is not None else label
            else:
                caption = captions[i]
            ax.text(x2+8, y2 + 8, caption,
                    color= color, size=20, backgroundcolor="none")


        # Mask
        if show_mask:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask  # 上下左右各有一行是空的
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1  # 横纵坐标颠倒
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)





    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
def display_instances_hospital(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(512, 512), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None,show_captions=True,file_path=''):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # if not N:
    #     print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax: #没有绘图的就用默认的
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    clip_boxes = []


    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        if show_bbox:#绘制矩形框
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="solid",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

            #放大区域
            # x1_f = max((x1+x2)/2-0.1*width,0)
            # y1_f = max((y1+y2)/2-0.1*height,0)
            # w_f = 0.2*width
            # h_f = 0.2*height
            # clip_boxes.append([x1_f,y1_f,w_f,h_f])
            # p1 = patches.Rectangle((x1_f, y1_f),w_f , h_f, linewidth=5,
            #                       alpha=1, linestyle="dashed",
            #                       edgecolor=color, facecolor='none')
            # ax.add_patch(p1)
            ax.add_patch(p)

        # Label
        if show_captions:
            if not captions:  # captions如果有值，表示框上面的字。没有就是label+score
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y2+8, caption,
                    color=color, size=25, backgroundcolor="none")


        # Mask
        if show_mask:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask  # 上下左右各有一行是空的
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1  # 横纵坐标颠倒
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)


    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
def display_instances_pingming(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(512, 512), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None,show_captions=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # if not N:
    #     print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax: #没有绘图的就用默认的
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    # colors = colors or random_colors(N)
    colors = ["red","orange","w"]
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        score = scores[i] if scores is not None else None
        if score is None:
            color = "red"
        elif score>0.95:
            color = colors[0]
        elif score>0.8:
            color =colors[1]
        else:
            color = colors[2]



        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        if show_bbox:#绘制矩形框
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if show_captions:
            if not captions:  # captions如果有值，表示框上面的字。没有就是label+score
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x2, y2 + 8, caption,
                    color=color, size=40, backgroundcolor="none")


        # Mask
        if show_mask:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask  # 上下左右各有一行是空的
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1  # 横纵坐标颠倒
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
def display_sem_seg(image, sem_seg, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,show_captions=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    N = len(class_ids)
    auto_show = False
    if not ax: #没有绘图的就用默认的
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Label
        if show_captions:
            if not captions:  # captions如果有值，表示框上面的字。没有就是label+score
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")


        # Mask
        mask = sem_seg
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask  #上下左右各有一行是空的
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1 #横纵坐标颠倒
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

def compute_ap_on_all_images(all_np_res, all_gt_res,iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match = []
    pred_match = []
    overlaps = []
    scores = []
    for np_res, gt_res in zip(all_np_res, all_gt_res):
        gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks = gt_res['boxes'], gt_res[
            'labels'], gt_res['masks'], np_res['boxes'], np_res['labels'], np_res['scores'], np_res['masks']
        g_m, p_m, overlap = compute_matches(
            gt_boxes, gt_class_ids, gt_masks,
            pred_boxes, pred_class_ids, pred_scores, pred_masks,
            iou_threshold)
        indices = np.argsort(pred_scores)[::-1]
        pred_scores = pred_scores[indices]
        scores.extend(pred_scores)
        gt_match.extend(g_m)
        pred_match.extend(p_m)
        overlaps.extend(overlap)
    #重新按照score排序
    pred_match = np.array(pred_match)
    gt_match = np.array(gt_match)
    overlaps = np.array(overlaps)
    indices = np.argsort(scores)[::-1]
    pred_match = pred_match[indices]

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def convert_instance_to_np(instance):
    res = {}
    res['boxes'] = instance.get("pred_boxes").tensor.to("cpu").numpy()[:, [1,0,3,2]]
    res['scores'] = instance.get("scores").to("cpu").numpy()
    res['labels'] = instance.get("pred_classes").to("cpu").numpy()
    if instance.has("pred_masks"):
        res['masks'] = instance.get("pred_masks").to("cpu").numpy().transpose(1,2,0)
    return res

def polygons_to_bitmask(polygons: List[np.ndarray], height: int=512, width: int=512) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool)

def convert_coco_format_to_np(coco_dict,is_set_category_id_to_zero=False):
    annotations = coco_dict['annotations']
    boxes = []
    seg_masks = []
    category_ids = []
    for ann in annotations:
        boxes.append([ann['bbox'][1], ann['bbox'][0], ann['bbox'][1]+ann['bbox'][3], ann['bbox'][0]+ann['bbox'][2]])
        if 'segmentation' in ann:
            seg_mask = polygons_to_bitmask(ann['segmentation'],width=coco_dict['width'],height=coco_dict['height'])#返回的mask是xy颠倒的
            seg_masks.append(seg_mask)

        if is_set_category_id_to_zero:
            category_ids.append(ann["category_id"]-1)
        else:
            category_ids.append(ann["category_id"])



    if len(seg_masks)>0:
        masks = np.array(seg_masks).transpose(1, 2, 0)
    else:
        masks = None
    res = {
        "boxes": np.array(boxes),
        "masks": masks,
        'labels': np.array(category_ids)
    }
    return res


def compute_DSC(m1, m2):
    masks1 = np.zeros(shape=(512, 512), dtype=np.bool)
    for i in range(m1.shape[-1]):
        masks1 = masks1 | m1[:, :, i]
    masks2 = np.zeros(shape=(512, 512), dtype=np.bool)
    for i in range(m2.shape[-1]):
        masks2 = masks2 | m2[:, :, i]
    # masks1 = masks1[..., None]
    # masks2 = masks2[..., None]
    # # flatten masks and compute their areas
    # masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    # masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1)
    area2 = np.sum(masks2)

    # intersections and union
    intersections = masks1 & masks2
    area3 = np.sum(intersections)
    # union = area1[:, None] + area2[None, :] - intersections

    dsc = (area3 * 2)/(area1+area2)

    return dsc
def compute_hospital_dice(np_res,gt_res):
    label_num = 2
    m1 = np_res["masks"]
    m2 = gt_res["masks"]
    masks1 = np.zeros(shape=(label_num,512, 512), dtype=np.bool)
    for i in range(m1.shape[-1]): #prediction
        class_id = np_res["labels"][i]
        masks1[class_id,:,:] = masks1[class_id,:,:] | m1[:, :, i]
    masks2 = np.zeros(shape=(label_num,512, 512), dtype=np.bool)
    for i in range(m2.shape[-1]): #gt
        class_id = gt_res["labels"][i]
        masks2[class_id,:,:] = masks2[class_id,:,:] | m2[:, :, i]

    dsc = []
    for i in range(masks1.shape[0]):
        area1 = np.sum(masks1[i,:,:])
        area2 = np.sum(masks2[i,:,:])

        # intersections and union
        intersections = masks1[i,:,:] & masks2[i,:,:]
        area3 = np.sum(intersections)
        # union = area1[:, None] + area2[None, :] - intersections
        if area1+area2 != 0:
            dsc.append((area3 * 2)/(area1+area2))
        else:
            dsc.append(np.nan)
    return  dsc
def compute_hospital_dice_bbox(np_res,gt_res):
    label_num = 2
    b1 = np.around(np_res["boxes"]).astype('uint')
    b2 = np.around(gt_res["boxes"]).astype('uint')


    m1 = np.zeros((512,512,b1.shape[0]),dtype=np.bool)
    m2 = np.zeros((512,512,b2.shape[0]),dtype=np.bool)

    for i in range(m1.shape[-1]):
        m1[b1[i,0]:b1[i,2],b1[i,1]:b1[i,3], i] = True
    for i in range(m2.shape[-1]):
        m2[b2[i,0]:b2[i,2],b2[i,1]:b2[i,3], i] = True
    # a = np_res["masks"][int((b1[i,0]+b2[i,2])/2),int((b1[i,1]+b2[i,3])/2),0]

    masks1 = np.zeros(shape=(label_num,512, 512), dtype=np.bool)
    for i in range(m1.shape[-1]):
        class_id = np_res["labels"][i]
        masks1[class_id,:,:] = masks1[class_id,:,:] | m1[:, :, i]
    masks2 = np.zeros(shape=(label_num,512, 512), dtype=np.bool)
    for i in range(m2.shape[-1]):
        class_id = gt_res["labels"][i]
        masks2[class_id,:,:] = masks2[class_id,:,:] | m2[:, :, i]

    DSC = []
    for i in range(masks1.shape[0]):
        area1 = np.sum(masks1[i,:,:])
        area2 = np.sum(masks2[i,:,:])

        # intersections and union
        intersections = masks1[i,:,:] & masks2[i,:,:]
        area3 = np.sum(intersections)
        # union = area1[:, None] + area2[None, :] - intersections
        if area1+area2 != 0:
            dsc = (area3 * 2)/(area1+area2)
        else:
            dsc = np.nan
        DSC.append(dsc)
    return  DSC

def compute_hospital_classification_box(np_res, gt_res):
    label_num = 2
    b1 = np.around(np_res["boxes"]).astype('uint')
    b2 = np.around(gt_res["boxes"]).astype('uint')

    m1 = np.zeros((512, 512, b1.shape[0]), dtype=np.bool)
    m2 = np.zeros((512, 512, b2.shape[0]), dtype=np.bool)

    for i in range(m1.shape[-1]):
        m1[b1[i,0]:b1[i,2],b1[i,1]:b1[i,3], i] = True
    for i in range(m2.shape[-1]):
        m2[b2[i,0]:b2[i,2],b2[i,1]:b2[i,3], i] = True

    labels = gt_res["labels"]

    np_counts = m1.shape[-1]
    gt_counts = m2.shape[-1]

    find_counts = 0
    correct_counts = np.zeros(4) #TP TN FP FN

    scores = np.zeros((np_counts,gt_counts))
    for i in range(np_counts):
        for j in range(gt_counts):
            area1 = np.sum(m1[:, :, i])
            area2 = np.sum(m2[:, :, j])
            area3 = np.sum(m1[:, :, i]&m2[:, :, j])
            scores[i][j] = (area3*2)/(area1+area2)
    threshold = 0.5
    if len(scores) == 0:
        print("为0了：",str(np_counts),"  ",str(gt_counts))
    while len(scores) and scores.max()>=threshold:
        find_counts += 1
        r,c = np.argmax(scores)//gt_counts,np.argmax(scores)%gt_counts

        if np_res["labels"][r] == 0 and gt_res["labels"][c] == 0:#TP
            correct_counts[0] += 1
        elif np_res["labels"][r] == 1 and gt_res["labels"][c] == 1:#TN
            correct_counts[1] += 1
        elif np_res["labels"][r] == 1 and gt_res["labels"][c] == 0:#FP
            correct_counts[2] += 1
        else:#FN
            correct_counts[3] += 1
        scores[:,c] = 0

    gt_counts = np.zeros(2)
    gt_counts[0] = len(labels) - sum(labels)
    gt_counts[1] = sum(labels)

    return  np_counts,gt_counts,find_counts,correct_counts

def compute_hospital_classification(np_res, gt_res):
    m1 = np_res["masks"]
    m2 = gt_res["masks"]

    labels = gt_res["labels"]

    np_counts = m1.shape[-1]
    gt_counts = m2.shape[-1]

    find_counts = 0
    correct_counts = np.zeros(4) #TP TN FP FN

    scores = np.zeros((np_counts,gt_counts))
    for i in range(np_counts):
        for j in range(gt_counts):
            area1 = np.sum(m1[:, :, i])
            area2 = np.sum(m2[:, :, j])
            area3 = np.sum(m1[:, :, i]&m2[:, :, j])
            scores[i][j] = (area3*2)/(area1+area2)
    threshold = 0.5
    if len(scores) == 0:
        print("为0了：",str(np_counts),"  ",str(gt_counts))
    while len(scores) and scores.max()>=threshold:
        find_counts += 1
        r,c = np.argmax(scores)//gt_counts,np.argmax(scores)%gt_counts

        if np_res["labels"][r] == 0 and gt_res["labels"][c] == 0:#TP
            correct_counts[0] += 1
        elif np_res["labels"][r] == 1 and gt_res["labels"][c] == 1:#TN
            correct_counts[1] += 1
        elif np_res["labels"][r] == 1 and gt_res["labels"][c] == 0:#FP
            correct_counts[2] += 1
        else:#FN
            correct_counts[3] += 1
        scores[:,c] = 0

    gt_counts = np.zeros(2)
    gt_counts[0] = len(labels) - sum(labels)
    gt_counts[1] = sum(labels)

    return  np_counts,gt_counts,find_counts,correct_counts

def calculate_f1(targets,predictions): #0 negative  1 positive
    TP, TN, FP, FN = 0, 0, 0, 0
    pre_results = np.array(predictions)
    tar_results = np.array(targets)
    TP = np.sum(pre_results & tar_results)
    TN = pre_results.shape[0] - np.sum(pre_results | tar_results)
    FP = np.sum((-1 * pre_results + 1) & tar_results)
    FN = np.sum((-1 * tar_results + 1) & pre_results)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print("TP:{},TN:{},FP:{},FN:{}".format(TP,TN,FP,FN))
    return precision,recall,F1



