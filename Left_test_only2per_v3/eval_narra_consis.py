import sys
sys.path.insert(0, r"./Depth-Anything-V2")
import os
import json
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.optimize import linear_sum_assignment

# ===================== 全局配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
METHODS = ["baseline1", "baseline2", "baseline3", "baseline4", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_baseline4", "_method"]
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "narrative_consistency_report.json"

# side_by_side 连续动作序列的三帧顺序
SEQUENCE_FRAMES = [
    "side_by_side_stand",
    "side_by_side_raise_hand",
    "side_by_side_point"
]

# ===================== 加载模型 =====================
print("Loading pose estimation model (YOLOv8n-Pose)...")
pose_model = YOLO("../yolov8n-pose.pt")
print("Pose model loaded successfully")
print("Loading depth model Depth Anything V2 (vitb)...")
depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 382, 768])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=DEVICE), strict=True)
depth_model = depth_model.to(DEVICE).eval()
print(f"Depth model loaded successfully, device: {DEVICE}")

# ===================== 关键点映射与GT对齐 =====================
# COCO 17点 → 自定义11点 映射（与数据集生成脚本完全对齐）
COCO_TO_CUSTOM = {
    0: 0,   # nose
    5: 1,   # left shoulder
    6: 2,   # right shoulder
    7: 3,   # left elbow
    8: 4,   # right elbow
    9: 5,   # left wrist
    10: 6,  # right wrist
    13: 9,  # left knee
    14: 10  # right knee
}


def get_gt_keypoints(scene_name, person_idx):
    """
    与 generate_synthetic_datasetu.py 中 get_scene_keypoints 完全对齐的GT关键点
    输入完整场景名，输出归一化坐标 [11, 2]
    """
    print(f"  [FUNC] get_gt_keypoints called, scene={scene_name}, person={person_idx}")
    # side_by_side 系列右侧人物完全一致
    p2_side_common = np.array([
        [0.70, 0.42], [0.64, 0.52], [0.76, 0.52],
        [0.60, 0.65], [0.80, 0.65], [0.58, 0.78], [0.82, 0.78],
        [0.70, 0.49], [0.70, 0.72], [0.64, 0.88], [0.76, 0.88]
    ])
    
    if scene_name == "side_by_side_stand":
        print("  [FUNC] matched scene: side_by_side_stand")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.40, 0.65], [0.18, 0.78], [0.42, 0.78],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        result = p1 if person_idx == 1 else p2_side_common
        print(f"  [FUNC] GT keypoints shape: {result.shape}")
        return result
    
    elif scene_name == "side_by_side_raise_hand":
        print("  [FUNC] matched scene: side_by_side_raise_hand")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.38, 0.45], [0.18, 0.78], [0.35, 0.38],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        result = p1 if person_idx == 1 else p2_side_common
        print(f"  [FUNC] GT keypoints shape: {result.shape}")
        return result
    
    elif scene_name == "side_by_side_point":
        print("  [FUNC] matched scene: side_by_side_point")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.46, 0.54], [0.18, 0.78], [0.56, 0.52],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        result = p1 if person_idx == 1 else p2_side_common
        print(f"  [FUNC] GT keypoints shape: {result.shape}")
        return result
    
    elif scene_name == "handshake":
        print("  [FUNC] matched scene: handshake")
        p1 = np.array([
            [0.35, 0.42], [0.29, 0.52], [0.41, 0.52],
            [0.25, 0.65], [0.48, 0.60], [0.22, 0.78], [0.50, 0.62],
            [0.35, 0.49], [0.35, 0.72], [0.30, 0.88], [0.40, 0.88]
        ])
        p2 = np.array([
            [0.65, 0.42], [0.59, 0.52], [0.71, 0.52],
            [0.52, 0.60], [0.75, 0.65], [0.50, 0.62], [0.78, 0.78],
            [0.65, 0.49], [0.65, 0.72], [0.60, 0.88], [0.70, 0.88]
        ])
        result = p1 if person_idx == 1 else p2
        print(f"  [FUNC] GT keypoints shape: {result.shape}")
        return result
    
    else:  # front_back
        print("  [FUNC] matched scene: front_back")
        p1 = np.array([
            [0.44, 0.42], [0.38, 0.52], [0.50, 0.52],
            [0.34, 0.65], [0.54, 0.65], [0.32, 0.78], [0.56, 0.78],
            [0.44, 0.49], [0.44, 0.72], [0.40, 0.88], [0.48, 0.88]
        ])
        p2 = np.array([
            [0.58, 0.35], [0.54, 0.42], [0.62, 0.42],
            [0.51, 0.52], [0.65, 0.52], [0.49, 0.62], [0.67, 0.62],
            [0.58, 0.40], [0.58, 0.58], [0.55, 0.72], [0.61, 0.72]
        ])
        result = p1 if person_idx == 1 else p2
        print(f"  [FUNC] GT keypoints shape: {result.shape}")
        return result


def compute_oks(gt_kpts, pred_kpts, sigma=0.1):
    """计算11点关键点相似度OKS"""
    print(f"  [FUNC] compute_oks called, sigma={sigma}")
    if len(gt_kpts) != len(pred_kpts):
        print("  [WARN] keypoint count mismatch, returning 0.0")
        return 0.0
    valid = (gt_kpts[:, 0] >= 0) & (pred_kpts[:, 0] >= 0)
    valid_count = np.sum(valid)
    print(f"  [FUNC] valid keypoints: {valid_count}")
    if valid_count < 3:
        print("  [WARN] too few valid keypoints, returning 0.0")
        return 0.0
    dist = np.linalg.norm(gt_kpts[valid] - pred_kpts[valid], axis=1)
    oks = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    result = float(np.mean(oks))
    print(f"  [FUNC] OKS result: {result:.4f}")
    return result


def compute_pose_fidelity(img_path, scene_name):
    """姿态保真度评估：返回双人平均OKS"""
    print(f"\n  [EVAL] Computing pose fidelity for: {os.path.basename(img_path)}")
    print(f"  [EVAL] scene: {scene_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        print("  [ERROR] Failed to load image, returning 0.0")
        return 0.0
    print(f"  [EVAL] Image loaded, shape: {img.shape}")
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    print(f"  [EVAL] Image resized to {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    results = pose_model(img, verbose=False)
    print(f"  [EVAL] YOLO pose detection completed, {len(results)} result(s)")
    
    gt_list = [get_gt_keypoints(scene_name, 1), get_gt_keypoints(scene_name, 2)]
    n_gt = len(gt_list)
    print(f"  [EVAL] GT persons count: {n_gt}")
    
    if len(results) == 0 or results[0].keypoints is None:
        print("  [WARN] No keypoints detected, returning 0.0")
        return 0.0
    
    h, w = img.shape[:2]
    det_kpts = results[0].keypoints.data.cpu().numpy()
    print(f"  [EVAL] Detected persons: {len(det_kpts)}")
    persons = []
    
    for person_idx, kp_data in enumerate(det_kpts):
        print(f"  [EVAL] Processing detected person {person_idx+1}")
        kp_17 = kp_data[:, :2] / np.array([w, h])
        conf = kp_data[:, 2]
        kp_11 = np.full((11, 2), -1.0, dtype=np.float32)
        
        # 直接映射关节点
        mapped_count = 0
        for coco_idx, cus_idx in COCO_TO_CUSTOM.items():
            if conf[coco_idx] > 0.5:
                kp_11[cus_idx] = kp_17[coco_idx]
                mapped_count += 1
        print(f"  [EVAL] Directly mapped keypoints: {mapped_count}")
        
        # 计算颈部（双肩中点）
        if conf[5] > 0.5 and conf[6] > 0.5:
            kp_11[7] = (kp_17[5] + kp_17[6]) / 2.0
            print("  [EVAL] Neck keypoint computed from shoulders")
        
        # 计算骨盆（双髋中点）
        if conf[11] > 0.5 and conf[12] > 0.5:
            kp_11[8] = (kp_17[11] + kp_17[12]) / 2.0
            print("  [EVAL] Pelvis keypoint computed from hips")
        
        persons.append(kp_11)
    
    n_det = len(persons)
    if n_det == 0:
        print("  [WARN] Zero valid persons after processing, returning 0.0")
        return 0.0
    
    # 匈牙利匹配最优对应
    print(f"  [EVAL] Running Hungarian matching (GT={n_gt}, Det={n_det})")
    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = -compute_oks(gt_list[i], persons[j], sigma=0.1)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    print(f"  [EVAL] Matched {len(row_ind)} pairs")
    
    total_oks = sum([-cost[r, c] for r, c in zip(row_ind, col_ind)])
    print(f"  [EVAL] Sum of matched OKS: {total_oks:.4f}")
    
    # 除以2（总人数），少检测到人物会被惩罚
    result = float(total_oks / n_gt)
    print(f"  [EVAL] Final average pose fidelity OKS: {result:.4f}")
    return result


# ===================== 深度与背景评估 =====================
def get_background_mask(mask_path):
    print(f"  [FUNC] get_background_mask called: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("  [WARN] mask file not found, returning all-white background mask")
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    bg_mask = (mask == 0).astype(np.uint8)
    print(f"  [FUNC] Background pixels: {np.sum(bg_mask)}")
    return bg_mask


def compute_background_ssim(img1, img2, bg_mask1, bg_mask2):
    """仅在共同背景区域计算SSIM，衡量跨帧背景稳定性"""
    print(f"  [FUNC] compute_background_ssim called")
    common_bg = (bg_mask1 & bg_mask2).astype(np.uint8)
    common_pixels = np.sum(common_bg)
    print(f"  [FUNC] Common background pixels: {common_pixels}")
    
    if common_pixels == 0:
        print("  [WARN] No common background area, returning 1.0")
        return 1.0
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    _, ssim_full = ssim(img1_gray, img2_gray, full=True, data_range=255)
    result = float(np.mean(ssim_full[common_bg > 0]))
    print(f"  [FUNC] Masked background SSIM: {result:.4f}")
    return result


def compute_cross_frame_depth_ssim(depth1, depth2):
    """计算两帧预测深度图之间的SSIM，衡量跨帧深度结构一致性"""
    print(f"  [FUNC] compute_cross_frame_depth_ssim called")
    # 各自归一化到[0,1]，消除相对深度绝对值偏移的影响，仅衡量结构一致性
    d1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
    d2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
    score = float(ssim(d1_norm, d2_norm, data_range=1.0))
    print(f"  [FUNC] Cross-frame depth SSIM: {score:.4f}")
    return score


def compute_depth_metrics(img, gt_depth_path):
    """深度评估：同时返回SSIM与RMSE，对齐提案要求"""
    print(f"\n  [EVAL] Computing depth metrics")
    print(f"  [EVAL] GT depth path: {gt_depth_path}")
    
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        print("  [ERROR] GT depth not found, returning default values")
        return 0.0, 1.0
    print(f"  [EVAL] GT depth loaded, shape: {gt_depth.shape}")
    
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  [EVAL] Image converted to RGB for depth inference")
    
    with torch.no_grad():
        pred_depth = depth_model.infer_image(img_rgb, input_size=IMAGE_SIZE)
    print(f"  [EVAL] Depth inference completed, shape: {pred_depth.shape}")
    
    # 统一归一化到[0,1]区间
    pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
    gt_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    print("  [EVAL] Both depth maps normalized to [0,1]")
    
    ssim_score = float(ssim(pred_norm, gt_norm, data_range=1.0))
    rmse = float(np.sqrt(np.mean((pred_norm - gt_norm) ** 2)))
    print(f"  [EVAL] Depth SSIM: {ssim_score:.4f}, Depth RMSE: {rmse:.4f}")
    
    return ssim_score, rmse


# ===================== 主函数 =====================
def main():
    print("\n=== Narrative Consistency & Layout Accuracy Evaluation ===")
    print("Metrics: Pose Fidelity (OKS), Depth Fidelity (SSIM/RMSE), Cross-frame Background SSIM, Cross-frame Depth SSIM")
    
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    print(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    print(f"Loaded {len(configs)} sample configs")
    
    # 提取所有「角色对 + 背景」组合，用于构建序列
    sequence_groups = set()
    for cfg in configs:
        if cfg["sequence_group"] == "side_by_side":
            key = (tuple(cfg["characters"]), cfg["background"])
            sequence_groups.add(key)
    sequence_groups = list(sequence_groups)
    print(f"Found {len(sequence_groups)} side_by_side sequence groups")
    
    report = {}
    
    for method_idx, (method, suffix) in enumerate(zip(METHODS, SUFFIXES), 1):
        print(f"\n{'='*60}")
        print(f"Evaluating method [{method_idx}/{len(METHODS)}]: {method.upper()}")
        print(f"File suffix: {suffix}")
        print(f"{'='*60}")
        
        pose_scores = []
        depth_ssim_scores = []
        depth_rmse_scores = []
        bg_consistency_scores = []
        depth_consistency_scores = []
        skipped_samples = 0
        
        # 1. 单帧布局精度评估（所有样本）
        print("\n  -- Phase 1: Single-frame layout accuracy evaluation --")
        for cfg_idx, cfg in enumerate(configs, 1):
            sample_id = cfg["sample_id"]
            scene = cfg["scene"]
            print(f"\n    [{cfg_idx}/{len(configs)}] Sample: {sample_id}")
            print(f"    Scene: {scene}")
            
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            
            if not os.path.exists(img_path):
                print(f"    [SKIP] Image not found: {img_path}")
                skipped_samples += 1
                continue
            print(f"    Found generated image")
            
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            # 姿态保真度
            print("    Computing pose fidelity...")
            pose_oks = compute_pose_fidelity(img_path, scene)
            pose_scores.append(pose_oks)
            print(f"    Pose OKS added to statistics: {pose_oks:.4f}")
            
            # 深度保真度
            print("    Computing depth fidelity...")
            gt_depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")
            d_ssim, d_rmse = compute_depth_metrics(img, gt_depth_path)
            depth_ssim_scores.append(d_ssim)
            depth_rmse_scores.append(d_rmse)
            print(f"    Depth metrics added: SSIM={d_ssim:.4f}, RMSE={d_rmse:.4f}")
        
        # 2. 叙事序列跨帧一致性评估（仅side_by_side三帧序列）
        print("\n  -- Phase 2: Cross-frame narrative consistency evaluation --")
        print(f"  Processing {len(sequence_groups)} sequence groups")
        
        for seq_idx, (chars, bg_name) in enumerate(sequence_groups, 1):
            print(f"\n    Sequence group [{seq_idx}/{len(sequence_groups)}]: {chars[0]} vs {chars[1]} | {bg_name}")
            frame_imgs = []
            frame_bg_masks = []
            frame_pred_depths = []
            
            for frame_scene in SEQUENCE_FRAMES:
                sample_id = f"{frame_scene}_{bg_name}_{chars[0]}_vs_{chars[1]}"
                img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
                mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
                
                if not os.path.exists(img_path):
                    print(f"      [SKIP] Frame missing: {frame_scene}")
                    continue
                
                print(f"      Loading frame: {frame_scene}")
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                bg_mask = get_background_mask(mask_path)
                
                # 推理当前帧的预测深度图
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    pred_depth = depth_model.infer_image(img_rgb, input_size=IMAGE_SIZE)
                
                frame_imgs.append(img)
                frame_bg_masks.append(bg_mask)
                frame_pred_depths.append(pred_depth)
            
            print(f"    Valid frames in sequence: {len(frame_imgs)}")
            
            # 相邻帧计算背景一致性
            if len(frame_imgs) > 1:
                print(f"    Computing adjacent frame background SSIM...")
                for i in range(len(frame_imgs) - 1):
                    print(f"      Frame {i} → Frame {i+1}")
                    bg_ssim = compute_background_ssim(
                        frame_imgs[i], frame_imgs[i+1],
                        frame_bg_masks[i], frame_bg_masks[i+1]
                    )
                    bg_consistency_scores.append(bg_ssim)
                    print(f"      Background SSIM: {bg_ssim:.4f}")
                
                # 相邻帧计算深度一致性
                print(f"    Computing adjacent frame depth SSIM...")
                for i in range(len(frame_pred_depths) - 1):
                    print(f"      Frame {i} → Frame {i+1}")
                    depth_ssim = compute_cross_frame_depth_ssim(
                        frame_pred_depths[i], frame_pred_depths[i+1]
                    )
                    depth_consistency_scores.append(depth_ssim)
                    print(f"      Depth SSIM: {depth_ssim:.4f}")
            else:
                print("    Not enough frames for consistency calculation")
        
        # 汇总统计
        print(f"\n  -- Aggregating results for {method.upper()} --")
        if not pose_scores:
            print(f"  No valid images found for {method}, skipping")
            continue
        
        avg_pose = round(np.mean(pose_scores), 4)
        avg_depth_ssim = round(np.mean(depth_ssim_scores), 4)
        avg_depth_rmse = round(np.mean(depth_rmse_scores), 4)
        avg_bg_consistency = round(np.mean(bg_consistency_scores), 4) if bg_consistency_scores else 0.0
        avg_depth_consistency = round(np.mean(depth_consistency_scores), 4) if depth_consistency_scores else 0.0
        overall_layout = round((avg_pose + avg_depth_ssim + (1 - avg_depth_rmse)) / 3, 4)
        
        print(f"  Total evaluated samples: {len(pose_scores)}")
        print(f"  Skipped samples: {skipped_samples}")
        print(f"  Pose Fidelity (OKS): {avg_pose:.4f}")
        print(f"  Depth SSIM: {avg_depth_ssim:.4f}")
        print(f"  Depth RMSE: {avg_depth_rmse:.4f}")
        print(f"  Cross-frame Background Consistency: {avg_bg_consistency:.4f}")
        print(f"  Cross-frame Depth Consistency: {avg_depth_consistency:.4f}")
        print(f"  Overall Layout Accuracy Score: {overall_layout:.4f}")
        
        report[method] = {
            "layout_accuracy": {
                "pose_fidelity_oks": avg_pose,
                "depth_ssim": avg_depth_ssim,
                "depth_rmse": avg_depth_rmse,
                "overall_layout_score": overall_layout
            },
            "narrative_consistency": {
                "cross_frame_background_ssim": avg_bg_consistency,
                "cross_frame_depth_ssim": avg_depth_consistency
            },
            "total_evaluated_samples": len(pose_scores),
            "total_sequence_pairs": len(bg_consistency_scores)
        }
    
    print(f"\n{'='*60}")
    print("All methods evaluated")
    print(f"Saving report to: {SAVE_REPORT}")
    
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print("Report saved successfully")
    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    print("[ENTRY] Script started")
    main()
    print("[EXIT] Script finished")