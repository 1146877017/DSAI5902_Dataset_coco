import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from depth_anything.dpt import DPT_DINOv2

# ===================== 配置 =====================
GT_BASE = r"../coco_multi_person/complete_samples_512"
GT_RAW = os.path.join(GT_BASE, "raw")
GT_OPENPOSE = os.path.join(GT_BASE, "openpose")  # GT姿态图路径
GT_DEPTH = os.path.join(GT_BASE, "depth")
GT_MASK = os.path.join(GT_BASE, "mask")
GEN_ROOT = r"controlnet_results"
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_NAMES = ["baseline1", "baseline2", "baseline3", "method"]
IMAGE_SIZE = 512
SAVE_PATH = r"quantitative_evaluation_report.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载已有模型
pose_model = YOLO("yolov8n-pose.pt").to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# 加载DepthAnything模型
depth_model = DPT_DINOv2(encoder='dinov2_vitb', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load("depth_anything_v2_vitb.pth", map_location=device), strict=False)
depth_model = depth_model.to(device).eval()

# ===================== PCKh（和GT姿态匹配） =====================
def extract_pose_keypoints(img_path):
    """从图像中提取两个人的关键点，返回坐标"""
    img = cv2.imread(img_path)
    results = pose_model(img, conf=0.3, verbose=False)
    if len(results[0].keypoints.xy) >= 2:
        return results[0].keypoints.xy.cpu().numpy()[:2]  # 取前两个人
    return None

def compute_pckh(gt_kps, gen_kps, threshold=0.5):
    """
    计算PCKh：以头部长度为阈值，关键点误差在阈值内的比例
    """
    if gt_kps is None or gen_kps is None:
        return 0.0
    gt_center = np.mean(gt_kps, axis=1)
    gen_center = np.mean(gen_kps, axis=1)
    if np.linalg.norm(gt_center[0] - gen_center[0]) > np.linalg.norm(gt_center[0] - gen_center[1]):
        gen_kps = gen_kps[::-1]  # 交换顺序
    
    # 计算头部长度
    head_length_gt = np.linalg.norm(gt_kps[:,3,:] - gt_kps[:,4,:], axis=1)  # 耳朵关键点索引
    head_length_gt = np.mean(head_length_gt) if head_length_gt.mean() > 0 else 50
    pckh_scores = []
    for person in range(2):
        errors = np.linalg.norm(gt_kps[person] - gen_kps[person], axis=1)
        valid = np.sum(errors < threshold * head_length_gt)
        pckh_scores.append(valid / len(errors))
    return round(np.mean(pckh_scores), 4)

# ===================== 用DepthAnything提取生成图深度 =====================
def get_generated_depth(gen_img):
    """从生成图中提取深度图"""
    img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB) / 255.0
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = depth_model(img_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear'
        ).squeeze().cpu().numpy()
    # 归一化
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def compute_depth_metrics(gen_img, gt_depth_path):
    """计算生成图深度与GT深度的RMSE/SSIM"""
    # 提取生成图深度
    gen_depth = get_generated_depth(gen_img)
    # 加载GT深度
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    # 计算指标
    rmse = np.sqrt(np.mean((gen_depth - gt_depth) ** 2))
    ssim_val = ssim(gen_depth, gt_depth, data_range=1.0)
    return round(rmse,4), round(ssim_val,4)

# ===================== 适配COCO的CLIP相似度 =====================
def compute_masked_content_clip(gen_img, gt_img, mask_path):
    """生成图与原图掩码区域的CLIP相似度"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = (mask > 0).astype(np.uint8)
    gen_masked = cv2.bitwise_and(gen_img, gen_img, mask=mask)
    gt_masked = cv2.bitwise_and(gt_img, gt_img, mask=mask)
    gen_feat = clip_model.encode_image(clip_preprocess(Image.fromarray(cv2.cvtColor(gen_masked,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device))
    gt_feat = clip_model.encode_image(clip_preprocess(Image.fromarray(cv2.cvtColor(gt_masked,cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device))
    return round(torch.cosine_similarity(gen_feat, gt_feat).item(),4)

# ===================== 主流程 =====================
def main():
    img_names = [f for f in os.listdir(GT_RAW) if f.endswith(('.jpg', '.png'))]
    final_report = {}
    for suffix, method in zip(METHOD_SUFFIX, METHOD_NAMES):
        metrics = {"pose_pckh": [], "depth_rmse": [], "depth_ssim": [], "content_clip_sim": [], "layout_correct":0}
        for img_name in img_names:
            base_name = os.path.splitext(img_name)[0]
            # 路径
            gen_path = os.path.join(GEN_ROOT, f"{base_name}{suffix}.png")
            gt_raw_path = os.path.join(GT_RAW, img_name)
            gt_pose_path = os.path.join(GT_OPENPOSE, img_name)
            gt_depth_path = os.path.join(GT_DEPTH, img_name)
            gt_mask_path = os.path.join(GT_MASK, f"{base_name}.png")
            
            # 读取图像
            gen_img = cv2.imread(gen_path)
            gt_raw = cv2.imread(gt_raw_path)
            if gen_img is None or gt_raw is None: continue
            
            # 1. 计算PCKh
            gt_kps = extract_pose_keypoints(gt_pose_path)
            gen_kps = extract_pose_keypoints(gen_path)
            metrics["pose_pckh"].append(compute_pckh(gt_kps, gen_kps))
            
            # 2. 计算深度指标
            rmse, ssim_val = compute_depth_metrics(gen_img, gt_depth_path)
            metrics["depth_rmse"].append(rmse)
            metrics["depth_ssim"].append(ssim_val)
            
            # 3. 计算内容一致性CLIP
            metrics["content_clip_sim"].append(compute_masked_content_clip(gen_img, gt_raw, gt_mask_path))
            
            # 4. 布局正确判定（PCKh>0.5 + SSIM>0.6）
            if metrics["pose_pckh"][-1]>0.5 and metrics["depth_ssim"][-1]>0.6:
                metrics["layout_correct"]+=1

        # 汇总
        total = len(img_names)
        final_report[method] = {
            "Pose PCKh@0.5": round(np.mean(metrics["pose_pckh"]),4),
            "Depth RMSE": round(np.mean(metrics["depth_rmse"]),4),
            "Depth SSIM": round(np.mean(metrics["depth_ssim"]),4),
            "Masked Content CLIP Similarity": round(np.mean(metrics["content_clip_sim"]),4),
            "Layout Accuracy": round(metrics["layout_correct"]/total,4)
        }
    # 保存
    with open(SAVE_PATH,'w',encoding='utf-8') as f: json.dump(final_report,f,indent=4,ensure_ascii=False)
    print("\n COCO定量评估完成！")

if __name__ == "__main__":
    main()