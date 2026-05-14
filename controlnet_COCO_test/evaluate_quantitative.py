import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from tqdm import tqdm

# ===================== 路径配置 =====================
GT_BASE = r"../coco_multi_person/complete_samples_512"
GT_RAW = os.path.join(GT_BASE, "raw")
GT_DEPTH = os.path.join(GT_BASE, "depth")
GT_MASK = os.path.join(GT_BASE, "mask")
GEN_ROOT = r"controlnet_results"
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_NAMES = ["baseline1", "baseline2", "baseline3", "method"]
IMAGE_SIZE = 512
SAVE_PATH = r"quantitative_evaluation_report.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("yolov8n-pose.pt").to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def compute_pose_pck(gen_img):
    results = pose_model(gen_img, conf=0.3, verbose=False)
    if results[0].keypoints is None: return 0.0
    pck_scores = [np.sum((k[:,0]>0)&(k[:,1]>0))/17 for k in results[0].keypoints.xy.cpu().numpy()]
    return np.mean(pck_scores) if pck_scores else 0.0

def compute_depth_metrics(gen_img, gt_depth_path):
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gen_depth = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    rmse = np.sqrt(np.mean((gen_depth - gt_depth) ** 2))
    ssim_val = ssim(gen_depth, gt_depth, data_range=1.0)
    return round(rmse,4), round(ssim_val,4)

def compute_masked_clip(gen_img, gt_img, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = (mask > 0).astype(np.uint8)
    gen_masked = cv2.bitwise_and(gen_img, gen_img, mask=mask)
    gt_masked = cv2.bitwise_and(gt_img, gt_img, mask=mask)
    gen_feat = clip_model.encode_image(clip_preprocess(Image.fromarray(cv2.cvtColor(gen_masked,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device))
    gt_feat = clip_model.encode_image(clip_preprocess(Image.fromarray(cv2.cvtColor(gt_masked,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device))
    return round(torch.cosine_similarity(gen_feat, gt_feat).item(),4)

def main():
    img_names = [f for f in os.listdir(GT_RAW) if f.endswith(('.jpg', '.png'))]
    final_report = {}
    for suffix, method in zip(METHOD_SUFFIX, METHOD_NAMES):
        metrics = {"pose_pck": [], "depth_rmse": [], "depth_ssim": [], "clip_sim": [], "layout_correct":0}
        for img_name in tqdm(img_names, desc=f"评估 {method}"):
            base_name = os.path.splitext(img_name)[0]
            gen_path = os.path.join(GEN_ROOT, f"{base_name}{suffix}.png")
            gen_img = cv2.imread(gen_path)
            gt_raw = cv2.imread(os.path.join(GT_RAW, img_name))
            if gen_img is None or gt_raw is None: continue

            metrics["pose_pck"].append(compute_pose_pck(gen_img))
            rmse, ssim_val = compute_depth_metrics(gen_img, os.path.join(GT_DEPTH, img_name))
            metrics["depth_rmse"].append(rmse)
            metrics["depth_ssim"].append(ssim_val)
            metrics["clip_sim"].append(compute_masked_clip(gen_img, gt_raw, os.path.join(GT_MASK, f"{base_name}.png")))
            if metrics["pose_pck"][-1]>0.8 and metrics["depth_ssim"][-1]>0.5: metrics["layout_correct"]+=1

        total = len(img_names)
        final_report[method] = {
            "Pose PCK": round(np.mean(metrics["pose_pck"]),4),
            "Depth RMSE": round(np.mean(metrics["depth_rmse"]),4),
            "Depth SSIM": round(np.mean(metrics["depth_ssim"]),4),
            "CLIP Sim": round(np.mean(metrics["clip_sim"]),4),
            "Layout Accuracy": round(metrics["layout_correct"]/total,4)
        }
    with open(SAVE_PATH,'w',encoding='utf-8') as f: json.dump(final_report,f,indent=4,ensure_ascii=False)
    print("\n  定量评估完成！")

if __name__ == "__main__":
    main()