import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from depth_anything.dpt import DPT_DINOv2

# ===================== 配置 =====================
GT_DEPTH_DIR = r"../coco_multi_person/complete_samples_512/depth"
GT_OPENPOSE_DIR = r"../coco_multi_person/complete_samples_512/openpose"
GEN_RESULT_DIR = r"controlnet_results"
GT_RAW_DIR = r"../coco_multi_person/complete_samples_512/raw"
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]
METHOD_NAMES = ["baseline1", "baseline2", "baseline3", "method"]
IMAGE_SIZE = 512
SAVE_REPORT = r"reverse_evaluation_report.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("yolov8n-pose.pt").to(device)
depth_model = DPT_DINOv2(encoder='dinov2_vitb', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load("depth_anything_v2_vitb.pth", map_location=device), strict=False)
depth_model = depth_model.to(device).eval()

# ===================== 工具函数 =====================
def check_person_num(img_path):
    """检查生成图中是否有至少2个人"""
    img = cv2.imread(img_path)
    res = pose_model(img, conf=0.3, verbose=False)
    return 1 if (res[0].keypoints is not None and len(res[0].keypoints.xy)>=2) else 0

def compute_pose_match(base_name, gen_img_path):
    """【修复 Bug 1】: 计算生成图与GT姿态的PCKh，必须从真实原图提取真值关键点"""
    gt_real_path = os.path.join(GT_RAW_DIR, f"{base_name}.jpg")
    
    # 统一使用 resize 512 的机制规避坐标错位
    img_gt = cv2.imread(gt_real_path)
    img_gen = cv2.imread(gen_img_path)
    if img_gt is None or img_gen is None: return 0
    
    img_gt = cv2.resize(img_gt, (IMAGE_SIZE, IMAGE_SIZE))
    img_gen = cv2.resize(img_gen, (IMAGE_SIZE, IMAGE_SIZE))
    
    gt_res = pose_model(img_gt, conf=0.3, verbose=False)[0]
    gen_res = pose_model(img_gen, conf=0.3, verbose=False)[0]
    
    if gt_res.keypoints is None or gen_res.keypoints is None: return 0
    
    gt_kps = gt_res.keypoints.xy.cpu().numpy()[:2]
    gen_kps = gen_res.keypoints.xy.cpu().numpy()[:2]
    
    if len(gt_kps) < 2 or len(gen_kps) < 2: return 0
    
    # 质心排序与匹配
    if np.linalg.norm(np.mean(gt_kps[0], axis=0) - np.mean(gen_kps[0], axis=0)) > np.linalg.norm(np.mean(gt_kps[0], axis=0) - np.mean(gen_kps[1], axis=0)):
        gen_kps = gen_kps[::-1]
        
    head_length = np.mean(np.linalg.norm(gt_kps[:,3,:] - gt_kps[:,4,:], axis=1)) or 50
    errors = np.linalg.norm(gt_kps - gen_kps, axis=2)
    pckh = np.mean(errors < 0.5 * head_length)
    return 1 if pckh > 0.5 else 0

def compute_depth_match(gen_img_path, gt_depth_path):
    """【修复 Bug 4】: 计算生成图与GT深度的SSIM，加入双线性插值保障分辨率对齐"""
    gen_img = cv2.imread(gen_img_path)
    img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)/255.0
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = depth_model(img_tensor)
        # 显式插值上采样，防止SSIM计算时尺寸不匹配报错
        gen_depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear'
        ).squeeze().cpu().numpy()
        gen_depth = (gen_depth - gen_depth.min())/(gen_depth.max() - gen_depth.min() + 1e-8)
        
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None: return 0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min())/(gt_depth.max() - gt_depth.min() + 1e-8)
    
    ssim_val = ssim(gen_depth, gt_depth, data_range=1.0)
    return 1 if ssim_val > 0.6 else 0

# ===================== 主流程 =====================
def main():
    img_names = [f for f in os.listdir(GT_DEPTH_DIR) if f.endswith(('.jpg','.png'))]
    report = {}
    for suffix, method in zip(METHOD_SUFFIX, METHOD_NAMES):
        valid = 0
        total = len(img_names)
        for img_name in img_names:
            base_name = os.path.splitext(img_name)[0]
            gen_path = os.path.join(GEN_RESULT_DIR, f"{base_name}{suffix}.png")
            gt_pose_path = os.path.join(GT_OPENPOSE_DIR, img_name)
            gt_depth_path = os.path.join(GT_DEPTH_DIR, img_name)
            num_ok = check_person_num(gen_path)
            pose_ok = compute_pose_match(gt_pose_path, gen_path)
            depth_ok = compute_depth_match(gen_path, gt_depth_path)
            if num_ok and pose_ok and depth_ok:
                valid +=1
        # 反向验证准确率
        acc = round(valid/total,4)
        report[method] = {
            "reverse_validation_accuracy": acc,
            "valid_samples": valid,
            "total_samples": total
        }
        print(f"{method}: 反向验证准确率 {acc:.2%}")
    # 保存
    with open(SAVE_REPORT,'w',encoding='utf-8') as f: json.dump(report,f,indent=4,ensure_ascii=False)
    print("\n 反向验证完成！")

if __name__ == "__main__":
    main()