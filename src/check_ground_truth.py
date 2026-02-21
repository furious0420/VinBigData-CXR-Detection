
import json
import os

def check_ground_truth(ann_file, image_names):
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Create image map
    image_map = {img['file_name']: img['id'] for img in data['images']}
    
    # Create category map
    cat_map = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create annotation map
    ann_map = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_map:
            ann_map[img_id] = []
        ann_map[img_id].append(cat_map[ann['category_id']])
    
    print(f"{'Image Name':<45} | {'Ground Truth Annotations'}")
    print("-" * 80)
    
    for name in image_names:
        img_id = image_map.get(name)
        if img_id is None:
            print(f"{name:<45} | Image not found in annotations")
            continue
        
        anns = ann_map.get(img_id, ["No finding (Normal)"])
        print(f"{name:<45} | {', '.join(set(anns))}")

if __name__ == "__main__":
    ann_files = [
        r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_train.json",
        r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_val.json"
    ]
    
    # Extract original filenames from the prediction filenames
    prediction_files = [
        "pred_0006e0a85696f6bb578e84fafa9a5607.jpg",
        "pred_00176f7e1b1cb835123f95960b9a9efd.jpg",
        "pred_0046f681f078851293c4e710c4466058.jpg",
        "pred_004d2bc2111d639f5e8441ced52d55cb.jpg",
        "pred_005be26a68485912e007a3703f43d60a.jpg",
        "pred_0076d6a1e3139927fd62459c54276c3c.jpg",
        "pred_00948e3e6acc03044af454fb9700ca60.jpg",
        "pred_009917a5aad749f5237c4cef18f0f235.jpg",
        "pred_009d837e29ba400e03856cf8d6a5b545.jpg",
        "pred_00b05e693202bb65a0c0ca7a0201495d.jpg"
    ]
    
    image_names = [f.replace("pred_", "") for f in prediction_files]
    
    for ann_file in ann_files:
        print(f"\nChecking: {os.path.basename(ann_file)}")
        check_ground_truth(ann_file, image_names)

