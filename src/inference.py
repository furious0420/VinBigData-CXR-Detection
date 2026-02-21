
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import get_model
from dataset import VinBigDataDataset
from PIL import Image

def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

CLASS_NAMES = [
    "Aortic enlargement", "Pleural thickening", "Pleural effusion", 
    "Cardiomegaly", "Lung Opacity", "Nodule/Mass", "Consolidation", 
    "Pulmonary fibrosis", "Infiltration", "Atelectasis", "Other lesion", 
    "ILD", "Pneumothorax", "Calcification"
]

def visualize_prediction(image, prediction, threshold=0.3, output_path=None):
    # Convert image to numpy
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.ascontiguousarray(image)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    boxes = prediction['boxes'].cpu().detach().numpy()
    scores = prediction['scores'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            # label is 1-indexed in many cases, or 0-indexed depending on how it was loaded.
            # In our dataset.py, it's the COCO category_id.
            class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"ID {label}"
            
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name}: {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved prediction to {output_path}")

def run_inference(model_path, image_dir, output_dir, num_images=10, threshold=0.3):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    num_classes = 15
    model = get_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files = image_files[:num_images]
    
    transform = get_transform()
    
    with torch.no_grad():
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).to(device)
            
            predictions = model([img_tensor])
            prediction = predictions[0]
            
            output_path = os.path.join(output_dir, f"pred_{img_file}")
            visualize_prediction(img_tensor, prediction, threshold, output_path)

if __name__ == "__main__":
    image_directory = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/images/val"
    model_checkpoint = r"d:/dlpro/checkpoints/model_epoch_9.pth"
    output_directory = r"d:/dlpro/predictions"
    
    if os.path.exists(model_checkpoint):
        run_inference(model_checkpoint, image_directory, output_directory, num_images=10, threshold=0.3)
    else:
        print(f"Model checkpoint not found at {model_checkpoint}. Run training first.")

