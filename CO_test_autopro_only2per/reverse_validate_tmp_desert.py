# 暂时不需要

import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from depth_anything_v2.dpt import DepthAnythingV2  # 保持与评估脚本一致的 V2 接口

# ===================== 配置 =====================
GT_DEPTH_DIR = r"co_dat_only2per/complete_samples_512_filter/depth"
GT_RAW_DIR = r"co_dat_only2per/complete_samples_512_filter/raw"

# 明确划分两个生成的模态文件夹
GEN_BG_DIR = r"results_all_pure_background"       # 模式一：纯背景
GEN_ORIG_DIR = r"results_all_original_image"     # 模式二：原图保持

METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_NAMES = ["baseline1", "baseline2", "baseline3", "method"]
IMAGE_SIZE = 512
SAVE_REPORT = r"reverse_eval_report.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("../yolov8n-pose.pt").to(device)

# 正确初始化 Depth Anything V2 
depth_model = DepthAnythingV2(encoder='vitb', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=device), strict=True)
depth_model = depth_model.to(device).eval()

# ===================== 工具函数 =====================
def check_pure_background_validity(img_path):
    """【负样本控制】验证纯背景图中是否成功擦除了所有人（检测到的人数应为 0）"""
    img = cv2.imread(img_path)
    if img is None: return False
    res = pose_model(img, conf=0.3, verbose=False)[0]
    
    if res.keypoints is None or len(res.keypoints.xy) == 0:
        return True
    
    # 过滤全零的无效关键点行，确保无残留人体
    valid_poses = [kps for kps in res.keypoints.xy.cpu().numpy() if np.any(kps > 0)]
    return len(valid_poses) == 0

def compute_pose_match(base_name, gen_orig_path):
    """【正样本控制】验证原图保持模式下的姿态精确度（PCKh > 0.5）"""
    gt_real_path = os.path.join(GT_RAW_DIR, f"{base_name}.jpg")
    img_gt = cv2.imread(gt_real_path)
    img_gen = cv2.imread(gen_orig_path)
    if img_gt is None or img_gen is None: return False
    
    img_gt = cv2.resize(img_gt, (IMAGE_SIZE, IMAGE_SIZE))
    img_gen = cv2.resize(img_gen, (IMAGE_SIZE, IMAGE_SIZE))
    
    gt_res = pose_model(img_gt, conf=0.3, verbose=False)[0]
    gen_res = pose_model(img_gen, conf=0.3, verbose=False)[0]
    
    if gt_res.keypoints is None or gen_res.keypoints is None: return False
    
    gt_kps = [k for k in gt_res.keypoints.xy.cpu().numpy() if np.any(k > 0)]
    gen_kps = [k for k in gen_res.keypoints.xy.cpu().numpy() if np.any(k > 0)]
    
    if len(gt_kps) < 2 or len(gen_kps) < 2: return False
    
    gt_kps = np.array(gt_kps[:2])
    gen_kps = np.array(gen_kps[:2])
    
    # 质心对齐排序
    if np.linalg.norm(np.mean(gt_kps[0], axis=0) - np.mean(gen_kps[0], axis=0)) > \
       np.linalg.norm(np.mean(gt_kps[0], axis=0) - np.mean(gen_kps[1], axis=0)):
        gen_kps = gen_kps[::-1]
        
    head_length = np.mean(np.linalg.norm(gt_kps[:,3,:] - gt_kps[:,4,:], axis=1)) or 50
    errors = np.linalg.norm(gt_kps - gen_kps, axis=2)
    pckh = np.mean(errors < 0.5 * head_length)
    return pckh > 0.5

def compute_depth_match(gen_bg_path, gt_depth_path):
    """【背景保持控制】验证纯背景图的几何空间结构是否与真实背景一致（SSIM > 0.6）"""
    gen_img = cv2.imread(gen_bg_path)
    if gen_img is None: return False
    img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)/255.0
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = depth_model(img_tensor)
        gen_depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear'
        ).squeeze().cpu().numpy()
        gen_depth = (gen_depth - gen_depth.min())/(gen_depth.max() - gen_depth.min() + 1e-8)
        
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None: return False
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min())/(gt_depth.max() - gt_depth.min() + 1e-8)
    
    ssim_val = ssim(gen_depth, gt_depth, data_range=1.0)
    return ssim_val > 0.6

# ===================== 主流程 =====================
def main():
    img_names = [f for f in os.listdir(GT_DEPTH_DIR) if f.endswith(('.jpg','.png'))]
    report = {}
    
    for suffix, method in zip(METHOD_SUFFIX, METHOD_NAMES):
        valid_joint = 0
        total_samples = 0
        
        # 增加子指标统计
        bg_erase_ok_cnt = 0
        pose_keep_ok_cnt = 0
        depth_consistency_cnt = 0
        
        for img_name in img_names:
            base_name = os.path.splitext(img_name)[0]
            
            # 分别获取两个模式下的图像路径
            gen_bg_path = os.path.join(GEN_BG_DIR, f"{base_name}{suffix}.png")
            gen_orig_path = os.path.join(GEN_ORIG_DIR, f"{base_name}{suffix}.png")
            gt_depth_path = os.path.join(GT_DEPTH_DIR, img_name)
            
            # 容错机制：跳过生成失败或缺失的样本
            if not os.path.exists(gen_bg_path) or not os.path.exists(gen_orig_path):
                continue
                
            total_samples += 1
            
            # 执行三维度反向验证
            bg_erase_ok = check_pure_background_validity(gen_bg_path)
            pose_ok = compute_pose_match(base_name, gen_orig_path)
            depth_ok = compute_depth_match(gen_bg_path, gt_depth_path)
            
            if bg_erase_ok: bg_erase_ok_cnt += 1
            if pose_ok: pose_keep_ok_cnt += 1
            if depth_ok: depth_consistency_cnt += 1
            
            # 跨模式联合一致性判定：三者同时满足才算完全解耦成功
            if bg_erase_ok and pose_ok and depth_ok:
                valid_joint += 1
                
        # 计算各项指标的成功率
        acc_joint = round(valid_joint / total_samples, 4) if total_samples > 0 else 0
        report[method] = {
            "joint_consistency_accuracy": acc_joint,
            "bg_erasure_success_rate": round(bg_erase_ok_cnt / total_samples, 4) if total_samples > 0 else 0,
            "pose_fidelity_success_rate": round(pose_keep_ok_cnt / total_samples, 4) if total_samples > 0 else 0,
            "depth_consistency_success_rate": round(depth_consistency_cnt / total_samples, 4) if total_samples > 0 else 0,
            "valid_samples": valid_joint,
            "total_samples": total_samples
        }
        
        print(f"\n[{method.upper()}] 评估完成:")
        print(f"  -> 联合一致性成功率 (Joint Acc): {acc_joint:.2%}")
        print(f"  -> 人体彻底擦除率 (BG Erasure): {report[method]['bg_erasure_success_rate']:.2%}")
        print(f"  -> 姿态精准保留率 (Pose Fidelity): {report[method]['pose_fidelity_success_rate']:.2%}")
        
    # 保存
    with open(SAVE_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n[+] 反向验证完成！报告已保存至 {SAVE_REPORT}")

if __name__ == "__main__":
    main()