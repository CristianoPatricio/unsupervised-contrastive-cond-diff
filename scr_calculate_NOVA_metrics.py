import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import pandas as pd
import ast
import re
from tqdm import tqdm
from PIL import Image
import os
import numpy as np

def sort_bboxes_by_area(bboxes):
    """
    Sorts a list of bounding boxes by descending area.
    
    Parameters:
    - bboxes: List of bounding boxes in the format [ymin, xmin, ymax, xmax]
    
    Returns:
    - List of bounding boxes sorted by descending area
    """
    def bbox_area(bbox):
        ymin, xmin, ymax, xmax = bbox
        height = max(0, ymax - ymin)
        width = max(0, xmax - xmin)
        return height * width

    sorted_bboxes = sorted(bboxes, key=bbox_area, reverse=True)
    return sorted_bboxes

def convert_yxyx_to_xyxy(box):
    """Convert from (y_min, x_min, y_max, x_max) to (x_min, y_min, x_max, y_max)"""
    return [box[1], box[0], box[3], box[2]]

def compute_iou(box1, box2):
    # x1_min, y1_min, x1_max, y1_max = box1
    # x2_min, y2_min, x2_max, y2_max = box2

    # This is the correct way of getting the coordinates
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin + 1)
    inter_h = max(0, inter_ymax - inter_ymin + 1)
    inter_area = inter_w * inter_h

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate_metrics(predictions, ground_truths):
    results = {}

    def calculate_ap_at_threshold(iou_thresh):
        aps = []
        tp_total = 0
        fp_total = 0
        total_gt = 0

        for image_id in ground_truths:
            gt_boxes = ground_truths[image_id]
            pred_boxes = predictions[image_id]
            matched_gt = set()
            tp = []
            fp = []

            for pred in pred_boxes:
                ious = [compute_iou(pred, gt) for gt in gt_boxes]
                max_iou = max(ious) if ious else 0
                max_iou_idx = np.argmax(ious) if ious else -1

                if max_iou >= iou_thresh and max_iou_idx not in matched_gt:
                    tp.append(1)
                    fp.append(0)
                    matched_gt.add(max_iou_idx)
                else:
                    tp.append(0)
                    fp.append(1)

            tp = np.array(tp)
            fp = np.array(fp)

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # AP = average of precision values at true positive positions
            ap = np.sum(precisions[tp == 1]) / (tp.sum() + 1e-6) if tp.sum() > 0 else 0.0
            aps.append(ap)

            tp_total += int(tp.sum())
            fp_total += int(fp.sum())
            total_gt += len(gt_boxes)

        return np.mean(aps), tp_total, fp_total, total_gt

    # mAP@0.3
    mAP30, TP30, FP30, GT30 = calculate_ap_at_threshold(0.3)
    # mAP@0.5
    mAP50, TP50, FP50, GT50 = calculate_ap_at_threshold(0.5)

    # mAP@[.50:.95]
    mAPs = []
    for iou in np.arange(0.5, 1.0, 0.05):
        ap, _, _, _ = calculate_ap_at_threshold(iou)
        mAPs.append(ap)
    mAP_50_95 = np.mean(mAPs)

    # Accuracy at IoU 0.5
    ACC50 = TP50 / GT50 if GT50 > 0 else 0.0

    results["mAP@30"] = mAP30
    results["mAP@50"] = mAP50
    results["mAP@50:95"] = mAP_50_95
    results["ACC@50"] = ACC50
    results["TP@30"] = TP30
    results["FP@30"] = FP30
    results["GT@30"] = GT30

    return results

def plot_gt_and_preds(image, bboxes_gt, bboxes_pred, method=None, img_ID=None):
    # Create a figure and display the image
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Add BBoxes
    for box in bboxes_gt:
        y1, x1, y2, x2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=6, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    # Plot Pred BBoxes
    for box in bboxes_pred:
        y1, x1, y2, x2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=6, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    plt.tight_layout()

    # Save fig
    if os.path.exists(method) is False:
        os.makedirs(method)
    plt.savefig(f"{method}/{img_ID}.png")

def convert_bbox_dict_to_list(bbox_dict):
    """
    Converts a dict with 'x', 'y', 'width', 'height' keys to list of boxes [x_min, y_min, x_max, y_max]
    """
    boxes = []
    for x, y, w, h in zip(bbox_dict['x'], bbox_dict['y'], bbox_dict['width'], bbox_dict['height']):
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        # boxes.append([x_min, y_min, x_max, y_max])
        boxes.append([y_min, x_min, y_max, x_max])
    return boxes

if __name__ == "__main__":

    # NOTE: Modify this path
    PREDS_PATH = "/path/to/logs/logs/runs/NOVA_benchmark/replace_with_the_folder/"
    
    NUM_BBOXES_TO_EVAL = 5 # The largest 5 bounding-boxes

    # Load CSV with NOVA dataset metadata
    df = pd.read_csv("Data/NOVA/NOVA_metadata.csv")

    mAP30 = []
    mAP50 = []
    mAP5095 = []
    ACC50 = []
    TP30 = []
    FP30 = []
    for fold in range(1):
        fold=3
        # Load numpy file with predictions
        np_preds = np.load(os.path.join(PREDS_PATH, f"NOVA_bboxes_dict_fold{fold}.npy"), allow_pickle=True).item()

        predictions = {}
        ground_truths = {}

        # Iterate over predictions
        counter_filtered_images = 0
        for k in tqdm(np_preds):
            
            # Filter by T2 axial MRI (check caption)
            caption = df.loc[df.filename == k]['caption'].values[0]
            if str(caption).__contains__("T2"):
                counter_filtered_images += 1

                # Get prediction and add it to dict
                # Convert predicted boxes to (x_min, y_min, x_max, y_max)
                pred_boxes = np_preds[k]

                # Choose how many bboxes to keep given its area.
                # <sorted_boxes> is a list of bounding boxes sorted by descending area
                sorted_boxes = sort_bboxes_by_area(pred_boxes)
                predictions[k] = sorted_boxes[:NUM_BBOXES_TO_EVAL]

                # Get ground truth bbox and add it to dict
                bbox_gold_string = df.loc[df.filename == k]['bbox_gold'].values[0]
                bbox_gold = ast.literal_eval(bbox_gold_string)
                gt_boxes = convert_bbox_dict_to_list(bbox_gold)
                ground_truths[k] = gt_boxes

                # NOTE: Plot image with bounding boxes. Uncomment for debugging.
                # if k == "case0086_002.png":
                #    plot_gt_and_preds(Image.open(os.path.join("/home/cristiano/datasets/NOVA/images", k)), gt_boxes, sorted_boxes[:NUM_BBOXES_TO_EVAL], method="Ours", img_ID=k.split(".")[0])
            
        # Evaluate
        print(f">>>> Metrics for fold {fold}")
        metrics = evaluate_metrics(predictions, ground_truths)
        mAP30.append(metrics["mAP@30"])
        mAP50.append(metrics["mAP@50"])
        mAP5095.append(metrics["mAP@50:95"])
        ACC50.append(metrics["ACC@50"])
        TP30.append(metrics["TP@30"])
        FP30.append(metrics["FP@30"])
        
        # Print metrics
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print(f"-"*50)

    # Print average metrics
    print(">>>> Average over Folds <<<<")
    print(f"No. of T2 images: {counter_filtered_images}")
    print(f"mAP@30: {np.mean(np.array(mAP30))*100:.2f}")
    print(f"mAP@50: {np.mean(np.array(mAP50))*100:.2f}")
    print(f"mAP@50:95: {np.mean(np.array(mAP5095))*100:.2f}")
    print(f"ACC@50: {np.mean(np.array(ACC50))*100:.2f}")
    print(f"TP@30: {int(np.mean(np.array(TP30)))}/{metrics['GT@30']}")
    print(f"FP@30: {int(np.mean(np.array(FP30)))}")