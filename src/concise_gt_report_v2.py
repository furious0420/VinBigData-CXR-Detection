
import json
import os

def get_labels(ann_file, target_names):
    if not os.path.exists(ann_file):
        return {}
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_name = {img['id']: img['file_name'] for img in data['images']}
    cat_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(cat_to_name[ann['category_id']])
    
    name_to_id = {v: k for k, v in id_to_name.items()}
    results = {}
    for name in target_names:
        if name in name_to_id:
            img_id = name_to_id[name]
            labels = img_id_to_anns.get(img_id, ["No finding (Normal)"])
            results[name] = list(set(labels))
    return results

if __name__ == "__main__":
    targets = [
        "0006e0a85696f6bb578e84fafa9a5607.jpg",
        "00176f7e1b1cb835123f95960b9a9efd.jpg",
        "0046f681f078851293c4e710c4466058.jpg",
        "004d2bc2111d639f5e8441ced52d55cb.jpg",
        "005be26a68485912e007a3703f43d60a.jpg",
        "0076d6a1e3139927fd62459c54276c3c.jpg",
        "00948e3e6acc03044af454fb9700ca60.jpg",
        "009917a5aad749f5237c4cef18f0f235.jpg",
        "009d837e29ba400e03856cf8d6a5b545.jpg",
        "00b05e693202bb65a0c0ca7a0201495d.jpg"
    ]
    
    train_file = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_train.json"
    val_file = r"d:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_val.json"
    
    train_res = get_labels(train_file, targets)
    val_res = get_labels(val_file, targets)
    
    all_res = {**train_res, **val_res}
    
    report_path = r"d:/dlpro/final_comparison_utf8.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Ground Truth Comparison Report\n")
        f.write("=" * 30 + "\n")
        for name in targets:
            labels = all_res.get(name, ["Not Found in JSON"])
            f.write(f"{name}: {', '.join(labels)}\n")
    
    print(f"Report saved to {report_path}")
