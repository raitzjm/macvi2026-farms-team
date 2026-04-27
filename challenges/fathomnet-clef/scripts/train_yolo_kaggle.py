#!/usr/bin/env python3
"""
YOLO training script for FathomNet-CLEF on Kaggle
Usage: python train_yolo_kaggle.py --data-path /kaggle/input/fathomnet-2026 --wandb-key YOUR_KEY
"""
import os
import argparse
import json
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, output_dir, split='train'):
    """Convert COCO annotations to YOLO format with proper class mapping."""
    with open(coco_json_path) as f:
        data = json.load(f)
    
    # Create category mapping (1-41 -> 0-31)
    cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {c['id']: i for i, c in enumerate(cats)}
    
    img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_dims = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    # Group annotations
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Create labels
    labels_dir = Path(output_dir) / 'labels' / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id, anns_list in img_to_anns.items():
        file_name = img_id_to_file[img_id]
        label_name = Path(file_name).stem + '.txt'
        label_path = labels_dir / label_name
        
        lines = []
        for ann in anns_list:
            x, y, w, h = ann['bbox']
            img_w, img_h = img_id_to_dims[img_id]
            x_center = (x + w/2) / img_w
            y_center = (y + h/2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            class_idx = cat_id_to_idx[ann['category_id']]
            lines.append(f'{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}')
        
        label_path.write_text('\n'.join(lines))
    
    return len(img_to_anns)

def create_yolo_yaml(data_root, output_yaml):
    """Create YOLO dataset YAML."""
    yaml_content = f"""path: {data_root}
train: train_set
val: test_set
nc: 32
names:
  0: amphipod
  1: anemone
  2: barnacle
  3: benthic worm
  4: bivalve
  5: black coral
  6: bony fish
  7: brittle star
  8: calycophoran siphonophore
  9: chiton
  10: crab
  11: feather star
  12: hydroid
  13: isopod
  14: jelly
  15: larvacean
  16: octopus
  17: physonect siphonophore
  18: pyrosome
  19: sea cucumber
  20: sea fan
  21: sea pen
  22: sea slug
  23: sea snail
  24: sea squirt
  25: sea star
  26: shrimp
  27: soft coral
  28: sponge
  29: squat lobster
  30: stony coral
  31: urchin
"""
    Path(output_yaml).write_text(yaml_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/kaggle/input/fathomnet-2026')
    parser.add_argument('--wandb-key', default=os.environ.get('WANDB_API_KEY', ''))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=480)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    
    os.environ['WANDB_API_KEY'] = args.wandb_key
    
    from ultralytics import YOLO
    
    # Create YAML
    yaml_path = '/tmp/fathomnet.yaml'
    create_yolo_yaml(args.data_path, yaml_path)
    
    print(f"Starting YOLO training for {args.epochs} epochs...")
    
    model = YOLO('yolo11n.pt')
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='/tmp/yolo_output',
        name='kaggle_train',
        exist_ok=True,
        device=0,
        entity='little_farms',
        project='fathomnet2026'
    )
    print("Training complete!")

if __name__ == '__main__':
    main()