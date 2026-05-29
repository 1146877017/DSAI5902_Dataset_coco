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
GT_DEPTH_DIR = os.path.join(GT_BASE, "original_image") # 校对深度图真实类目名

# 直接读取上一轮实验生成的清单
MANIFEST_PATH = "eval_manifest_1201_1300.json" 
IMAGE_SIZE = 512
SAVE_PATH = r"quantitative_evaluation_report.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("yolov8n-pose.pt").to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

depth_model = DPT_DINOv2(encoder='dinov2_vitb', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load("depth_anything_v2_vitb.pth", map_location=device), strict=False)
depth_model = depth_model.to(device).eval()

def extract_pose_keypoints(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    # 【修复 Bug 3】: 强制缩放到统一尺寸，确保真值图与生成图的关键点在同一坐标系
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    results = pose_model(img, conf=0.3, verbose=False)
    
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None
    kps = results[0].keypoints.xy.cpu().numpy()
    if len(kps) >= 2:
        return kps[:2]
    return None

def compute_pckh(gt_kps, gen_kps, threshold=0.5):
    # 严格的边界守卫，防御 Baseline 崩溃
    if gt_kps is None or gen_kps is None or len(gt_kps) < 2 or len(gen_kps) < 2:
        return 0.0
    
    gt_center = np.mean(gt_kps, axis=1)
    gen_center = np.mean(gen_kps, axis=1)
    if np.linalg.norm(gt_center[0] - gen_center[0]) > np.linalg.norm(gt_center[0] - gen_center[1]):
        gen_kps = gen_kps[::-1]
    
    head_length_gt = np.mean(np.linalg.norm(gt_kps[:,3,:] - gt_kps[:,4,:], axis=1))
    head_length_gt = head_length_gt if head_length_gt > 0 else 50
    
    pckh_scores = []
    for person in range(2):
        errors = np.linalg.norm(gt_kps[person] - gen_kps[person], axis=1)
        valid = np.sum(errors < threshold * head_length_gt)
        pckh_scores.append(valid / len(errors))
    return round(np.mean(pckh_scores), 4)

def get_generated_depth(gen_img):
    img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB) / 255.0
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = depth_model(img_tensor)
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear').squeeze().cpu().numpy()
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def compute_depth_metrics(gen_img, gt_depth_path):
    gen_depth = get_generated_depth(gen_img)
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None: return 1.0, 0.0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    
    rmse = np.sqrt(np.mean((gen_depth - gt_depth) ** 2))
    ssim_val = ssim(gen_depth, gt_depth, data_range=1.0)
    return round(rmse,4), round(ssim_val,4)

def compute_split_person_clip(gen_img, mask_path, text_p1, text_p2):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return 0.0
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    
    person_ids = [p for p in np.unique(mask) if p > 0]
    centroids_x = []
    valid_ids = []
    for p in person_ids:
        inst_m = (mask == p).astype(np.uint8)
        M = cv2.moments(inst_m)
        if M["m00"] > 0:
            centroids_x.append(int(M["m10"] / M["m00"]))
            valid_ids.append(p)
            
    sorted_ids = [p for _, p in sorted(zip(centroids_x, valid_ids))][:2]
    if len(sorted_ids) < 2: return 0.0
    
    text_tokens = clip.tokenize([text_p1, text_p2]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    sim_scores = []
    for idx, p_id in enumerate(sorted_ids):
        p_mask = (mask == p_id).astype(np.uint8)
        # 提取生成图里的指定角色区域
        gen_masked = cv2.bitwise_and(gen_img, gen_img, mask=p_mask)
        
        # 【修复 Bug 2】: 计算Bounding Box，将目标人体紧密裁剪出来，防止CLIP缩放后特征缩水
        x, y, w, h = cv2.boundingRect(p_mask)
        if w > 0 and h > 0:
            gen_cropped = gen_masked[y:y+h, x:x+w]
        else:
            gen_cropped = gen_masked
            
        gen_pil = Image.fromarray(cv2.cvtColor(gen_cropped, cv2.COLOR_BGR2RGB))
        gen_tensor = clip_preprocess(gen_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feature = clip_model.encode_image(gen_tensor)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            
        sim = torch.cosine_similarity(img_feature, text_features[idx:idx+1]).item()
        sim_scores.append(sim)
        
    return round(np.mean(sim_scores), 4)

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f" 找不到本轮实验清单文件: {MANIFEST_PATH}，请确认路径。")
        return
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    METHODS = ["baseline1", "baseline2", "baseline3", "method"]
    # 建立后缀映射字典
    suffix_map = {
        "baseline1": "_baseline1.png",
        "baseline2": "_baseline2.png",
        "baseline3": "_baseline3.png",
        "method": "_method.png"
    }
    
    final_report = {}
    
    for method in METHODS:
        metrics = {"pose_pckh": [], "depth_rmse": [], "depth_ssim": [], "clip_text_sim": [], "layout_correct": 0}
        
        # 针对清单里的有效样本精准遍历
        for item in manifest_data:
            sample_name = item["sample_name"]
            mode = item["mode"]
            
            # 定位当前实验模式生成的图片夹
            output_dir = f"results_1201to1300_{mode}" 
            gen_path = os.path.join(output_dir, f"{sample_name}{suffix_map[method]}")
            
            gt_raw_path = os.path.join(GT_RAW, f"{sample_name}.jpg")
            gt_depth_path = os.path.join(GT_DEPTH_DIR, f"{sample_name}.png") # 动态适配原图/纯背景深度
            gt_mask_path = os.path.join(GT_BASE, "mask", f"{sample_name}.png")
            
            gen_img = cv2.imread(gen_path)
            if gen_img is None: continue
            
            # 1. 提取真值图像与生成图像的 Pose 关键点进行对比
            gt_kps = extract_pose_keypoints(gt_raw_path) # 修正：必须从RAW真实图提取真值关键点
            gen_kps = extract_pose_keypoints(gen_path)
            metrics["pose_pckh"].append(compute_pckh(gt_kps, gen_kps))
            
            # 2. 深度指标
            rmse, ssim_val = compute_depth_metrics(gen_img, gt_depth_path)
            metrics["depth_rmse"].append(rmse)
            metrics["depth_ssim"].append(ssim_val)
            
            # 3. 语义隔离与保持指标（图文匹配）
            clip_sim = compute_split_person_clip(gen_img, gt_mask_path, item["person1_gt"], item["person2_gt"])
            metrics["clip_text_sim"].append(clip_sim)
            
            # 4. 判定
            if metrics["pose_pckh"][-1] > 0.5 and metrics["depth_ssim"][-1] > 0.6:
                metrics["layout_correct"] += 1
                
        total = len(metrics["pose_pckh"]) if len(metrics["pose_pckh"]) > 0 else 1
        final_report[method] = {
            "Pose PCKh@0.5": round(np.mean(metrics["pose_pckh"]), 4),
            "Depth RMSE": round(np.mean(metrics["depth_rmse"]), 4),
            "Depth SSIM": round(np.mean(metrics["depth_ssim"]), 4),
            "Masked Text-Image CLIP Similarity": round(np.mean(metrics["clip_text_sim"]), 4),
            "Layout Accuracy": round(metrics["layout_correct"] / total, 4)
        }
        print(f">> {method} 评测完毕。布局准确率: {final_report[method]['Layout Accuracy']:.2%}")
        
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"\n COCO定量评估全面完成！报告已保存至 `{SAVE_PATH}`")

if __name__ == "__main__":
    main()