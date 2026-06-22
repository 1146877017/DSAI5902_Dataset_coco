import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# ===================== 路径配置 =====================
GT_RAW_DIR = r"../coco_multi_person/complete_samples_512/raw"
GEN_RESULT_DIR = r"controlnet_results"  
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_LABELS = ["Baseline 1", "Baseline 2", "Baseline 3", "Method"]
SAVE_VIS_DIR = r"comparison_visualize"
IMAGE_SIZE = 512
SAMPLE_NUM = 20
os.makedirs(SAVE_VIS_DIR, exist_ok=True)

def add_label(img, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (0, 0), (IMAGE_SIZE, 40), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
    return img

def main():
    img_names = [f for f in os.listdir(GT_RAW_DIR) if f.endswith(('.jpg', '.png'))]
    random.seed(42)
    sample_names = random.sample(img_names, min(SAMPLE_NUM, len(img_names)))
    
    for img_name in tqdm(sample_names, desc="生成对比图"):
        gt_path = os.path.join(GT_RAW_DIR, img_name)
        gt_img = cv2.imread(gt_path)
        if gt_img is None: continue
        gt_img = cv2.resize(gt_img, (IMAGE_SIZE, IMAGE_SIZE))
        gt_img = add_label(gt_img, "Input Image")

        gen_imgs = [gt_img]
        base_name = os.path.splitext(img_name)[0]
        
        for suffix, label in zip(METHOD_SUFFIX, METHOD_LABELS):
            gen_path = os.path.join(GEN_RESULT_DIR, f"{base_name}{suffix}.png")
            gen_img = cv2.imread(gen_path)
            
            if gen_img is None:
                gen_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            gen_img = cv2.resize(gen_img, (IMAGE_SIZE, IMAGE_SIZE))
            gen_img = add_label(gen_img, label)
            gen_imgs.append(gen_img)

        concat_img = np.hstack(gen_imgs)
        save_path = os.path.join(SAVE_VIS_DIR, f"compare_{img_name}")
        cv2.imwrite(save_path, concat_img)

    print("\n 对比图生成完成！")

if __name__ == "__main__":
    main()