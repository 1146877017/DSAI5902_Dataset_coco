import os
import json
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

# ===================== 路径配置 =====================
GT_DEPTH_DIR = r"../coco_multi_person/complete_samples_512/depth"
GEN_RESULT_DIR = r"controlnet_results"
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_NAMES = ["baseline1", "baseline2", "baseline3", "method"]
IMAGE_SIZE = 512
SAVE_REPORT = r"reverse_evaluation_report.json"

pose_model = YOLO("yolov8n-pose.pt")

def check_person_num(img):
    res = pose_model(img, conf=0.3, verbose=False)
    return 1.0 if (res[0].keypoints is not None and len(res[0].keypoints.xy)>=2) else 0.0

def main():
    img_names = [f for f in os.listdir(GT_DEPTH_DIR) if f.endswith(('.jpg','.png'))]
    report = {}
    for suffix, method in zip(METHOD_SUFFIX, METHOD_NAMES):
        correct = 0
        total = len(img_names)
        for img_name in img_names:
            base_name = os.path.splitext(img_name)[0]
            gen_path = os.path.join(GEN_RESULT_DIR, f"{base_name}{suffix}.png")
            img = cv2.imread(gen_path)
            if img is None: continue
            if check_person_num(img) == 1.0: correct +=1
        report[method] = {"Layout Accuracy": round(correct/total,4)}
        print(f"{method}: 布局准确率 {correct/total:.2%}")
    with open(SAVE_REPORT,'w',encoding='utf-8') as f: json.dump(report,f,indent=4,ensure_ascii=False)
    print("\n 反向验证完成！")

if __name__ == "__main__":
    main()