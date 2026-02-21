
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from dataset import VinBigDataDataset

def visualize_sample(image, target, output_path):
    # Image is a PIL Image, convert to numpy
    image = np.array(image)
    
    # Target contains 'boxes' and 'labels'
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    root_dir = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/images/train"
    ann_file = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_train.json"

    if not os.path.exists(root_dir):
        print(f"Image directory not found: {root_dir}")
        exit(1)
    
    dataset = VinBigDataDataset(root=root_dir, annotation_file=ann_file)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualizing the first sample
    img, target = dataset[2]
    visualize_sample(img, target, "d:/dlpro/sample_vis_2.png")
    
    # Visualizing another sample
    img, target = dataset[3]
    visualize_sample(img, target, "d:/dlpro/sample_vis_3.png")
