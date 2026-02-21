"""
COCO-style mAP evaluation using pycocotools.
"""
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(model, data_loader, device, annotation_file):
    """
    Evaluate model using COCO mAP metrics.
    
    Args:
        model: Trained Faster R-CNN model.
        data_loader: DataLoader for the validation set.
        device: torch device.
        annotation_file: Path to COCO-format annotation JSON.
    
    Returns:
        dict with AP metrics.
    """
    model.eval()
    coco_gt = COCO(annotation_file)

    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()   # [N, 4] in x1,y1,x2,y2
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],   # COCO expects [x, y, w, h]
                        "score": float(score),
                    })

    if len(results) == 0:
        print("No predictions made — skipping mAP computation.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "AP@[0.50:0.95]", "AP@0.50", "AP@0.75",
        "AP-S", "AP-M", "AP-L",
        "AR@1", "AR@10", "AR@100",
        "AR-S", "AR-M", "AR-L",
    ]
    metrics = {name: float(val) for name, val in zip(metric_names, coco_eval.stats)}
    return metrics
