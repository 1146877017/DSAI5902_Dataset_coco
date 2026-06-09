import os
import json
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
SCENES = ["side_by_side", "handshake", "front_back"]
METHODS = ["baseline1", "baseline2", "baseline3", "ablation1", "ablation2", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_ablation1", "_ablation2", "_method"]
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "narrative_consistency_report.json"

print(" 加载姿态估计模型 (YOLOv8-Pose) ...")
pose_model = YOLO("../yolov8n-pose.pt")

print(" 加载深度模型 Depth Anything V2 (vitb) ...")
depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=DEVICE), strict=True)
depth_model = depth_model.to(DEVICE).eval()

def get_background_mask(mask_path):
    """获取背景掩码（人物区域=0，背景=1）"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    bg_mask = (mask == 0).astype(np.uint8)
    return bg_mask

def compute_masked_ssim(img1_gray, img2_gray, mask):
    """只在掩码区域计算 SSIM，避免黑色填充导致虚高"""
    # 提取有效区域像素
    masked_img1 = img1_gray[mask > 0]
    masked_img2 = img2_gray[mask > 0]
    if len(masked_img1) < 2:
        return 0.0
    # 计算 SSIM
    # 掩码外区域用各自图像的均值填充
    img1_filled = img1_gray.copy()
    img2_filled = img2_gray.copy()
    bg_val1 = np.mean(img1_gray[mask > 0]) if np.any(mask) else 128
    bg_val2 = np.mean(img2_gray[mask > 0]) if np.any(mask) else 128
    img1_filled[mask == 0] = bg_val1
    img2_filled[mask == 0] = bg_val2
    return ssim(img1_filled, img2_filled, data_range=255)

def compute_background_consistency(img1, img2, mask1, mask2):
    """计算两个图像的背景一致性，使用各自的背景掩码"""
    # 使用两个掩码的交集区域进行评估（只有两帧都是背景的区域才比较）
    common_bg = (mask1 & mask2).astype(np.uint8)
    if np.sum(common_bg) == 0:
        return 0.0
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return compute_masked_ssim(img1_gray, img2_gray, common_bg)

def compute_depth_fidelity(img, gt_depth_path):
    """计算生成图像深度图与 GT 深度图之间的 SSIM"""
    gen_depth = depth_model.infer_image(img, input_size=IMAGE_SIZE)
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        return 0.0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    return ssim(gen_depth, gt_depth, data_range=1.0)

def compute_pose_fidelity(img, gt_pose_path):
    """计算生成图像与 GT 姿态图的结构相似度"""
    # GT 姿态图是单通道线条图，生成图像需转为灰度
    gen_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gt_pose = cv2.imread(gt_pose_path, cv2.IMREAD_GRAYSCALE)
    if gt_pose is None:
        return 0.0
    gt_pose = cv2.resize(gt_pose, (IMAGE_SIZE, IMAGE_SIZE))
    return ssim(gen_gray, gt_pose, data_range=255)

def main():
    print("\n【叙事一致性评估】")
    print("评估指标：跨场景背景一致性、深度保真度、姿态保真度")
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"错误：找不到 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # 提取所有角色对
    character_pairs = set()
    for cfg in configs:
        character_pairs.add((cfg["characters"][0], cfg["characters"][1]))
    character_pairs = list(character_pairs)
    print(f"检测到 {len(character_pairs)} 组角色配对")

    report = {}
    for method, suffix in zip(METHODS, SUFFIXES):
        print(f"\n>> 评估方法: {method.upper()}")

        # 存储各项指标
        bg_scores = []      # 背景一致性（跨场景）
        depth_scores = []   # 深度保真度（与 GT 对比）
        pose_scores = []    # 姿态保真度（与 GT 对比）

        for char1, char2 in character_pairs:
            # 收集该角色对在三个场景下的图像、掩码、GT路径
            seq_imgs = []
            seq_masks = []
            for scene in SCENES:
                sample_id = f"{scene}_{char1}_vs_{char2}"
                img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
                mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
                gt_depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")
                gt_pose_path = os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")
                if not os.path.exists(img_path):
                    break
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                seq_imgs.append(img)
                seq_masks.append(get_background_mask(mask_path))
                # 深度保真度：每个样本独立计算与 GT 深度图的相似度
                depth_score = compute_depth_fidelity(img, gt_depth_path)
                depth_scores.append(depth_score)
                # 姿态保真度：每个样本独立计算与 GT 姿态图的相似度
                pose_score = compute_pose_fidelity(img, gt_pose_path)
                pose_scores.append(pose_score)

            if len(seq_imgs) != len(SCENES):
                continue  # 该角色对缺失某些场景

            # 背景一致性：相邻场景之间背景区域的 SSIM
            # 只计算两帧共同背景区域，避免黑色填充干扰
            for i in range(len(seq_imgs)-1):
                bg = compute_background_consistency(seq_imgs[i], seq_imgs[i+1],
                                                    seq_masks[i], seq_masks[i+1])
                bg_scores.append(bg)

        # 汇总平均
        final_bg = round(float(np.mean(bg_scores)), 4) if bg_scores else 0.0
        final_depth = round(float(np.mean(depth_scores)), 4) if depth_scores else 0.0
        final_pose = round(float(np.mean(pose_scores)), 4) if pose_scores else 0.0
        final_total = round((final_bg + final_depth + final_pose) / 3, 4)

        print(f"    背景一致性（跨场景）: {final_bg:.4f}")
        print(f"    深度保真度（与GT对比）: {final_depth:.4f}")
        print(f"    姿态保真度（与GT对比）: {final_pose:.4f}")
        print(f"    总体叙事一致性: {final_total:.4f}")

        report[method] = {
            "background_consistency": final_bg,
            "depth_fidelity": final_depth,
            "pose_fidelity": final_pose,
            "overall_narrative_consistency": final_total,
            "total_evaluated_sequences": len(character_pairs)
        }

    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n叙事一致性评估完成，结果保存至 {SAVE_REPORT}")

if __name__ == "__main__":
    main()