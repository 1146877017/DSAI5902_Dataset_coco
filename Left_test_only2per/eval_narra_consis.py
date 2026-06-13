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
SCENES = ["side_by_side", "handshake", "front_back"]

# 支持的标准方法列表
METHODS = ["baseline1", "baseline2", "baseline3", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_method"]

IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "narrative_consistency_report.json"

# ===================== 加载模型 =====================
print(" 加载姿态估计模型 (YOLOv8-Pose) ...")
pose_model = YOLO("../yolov8n-pose.pt")

print(" 加载深度模型 Depth Anything V2 (vitb) ...")
depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=DEVICE), strict=True)
depth_model = depth_model.to(DEVICE).eval()

# ===================== 关键点定义与映射 =====================
# 移除 7(颈部) 和 8(骨盆) 的直接映射，改为在代码中通过相关关节点求中点合成
CUSTOM_TO_COCO = {
    0: 0,   # nose
    1: 5,   # left shoulder
    2: 6,   # right shoulder
    3: 7,   # left elbow
    4: 8,   # right elbow
    5: 9,   # left wrist
    6: 10,  # right wrist
    9: 13,  # left knee
    10: 14  # right knee
}

def get_gt_keypoints(scene, person_idx):
    """返回场景标准关键点（与 generate_synthetic_dataset_tmp.py 中的 get_scene_keypoints 一致）"""
    if scene == "side_by_side":
        p1 = np.array([[0.30,0.50],[0.28,0.58],[0.32,0.58],[0.26,0.68],[0.34,0.68],
                       [0.24,0.78],[0.36,0.78],[0.30,0.60],[0.30,0.70],[0.28,0.80],[0.32,0.80]])
        p2 = np.array([[0.70,0.50],[0.68,0.58],[0.72,0.58],[0.66,0.68],[0.74,0.68],
                       [0.64,0.78],[0.76,0.78],[0.70,0.60],[0.70,0.70],[0.68,0.80],[0.72,0.80]])
    elif scene == "handshake":
        p1 = np.array([[0.35,0.48],[0.30,0.55],[0.40,0.55],[0.27,0.65],[0.45,0.60],
                       [0.25,0.75],[0.50,0.62],[0.35,0.60],[0.35,0.72],[0.32,0.85],[0.38,0.85]])
        p2 = np.array([[0.65,0.48],[0.60,0.55],[0.70,0.55],[0.55,0.60],[0.73,0.65],
                       [0.50,0.62],[0.75,0.75],[0.65,0.60],[0.65,0.72],[0.62,0.85],[0.67,0.85]])
    else:  # front_back
        p1 = np.array([[0.44,0.42],[0.38,0.52],[0.50,0.52],[0.34,0.65],[0.54,0.65],
                       [0.32,0.78],[0.56,0.78],[0.44,0.49],[0.44,0.72],[0.40,0.88],[0.48,0.88]])
        p2 = np.array([[0.58,0.35],[0.54,0.42],[0.62,0.42],[0.51,0.52],[0.65,0.52],
                       [0.49,0.62],[0.67,0.62],[0.58,0.40],[0.58,0.58],[0.55,0.72],[0.61,0.72]])
    return p1 if person_idx == 1 else p2

def compute_oks(gt_kpts, pred_kpts, sigma=0.1):
    """ 计算自定义11点 OKS (Object Keypoint Similarity) """
    if len(gt_kpts) != len(pred_kpts):
        return 0.0
    valid = (gt_kpts[:,0] >= 0) & (pred_kpts[:,0] >= 0)
    if np.sum(valid) < 3:
        return 0.0
    dist = np.linalg.norm(gt_kpts[valid] - pred_kpts[valid], axis=1)
    oks = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    return float(np.mean(oks))

def compute_pose_fidelity(img_path, scene):
    """ 姿态保真度核心评估（含漏检/少人惩罚机制） """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    results = pose_model(img, verbose=False)
    
    # 两个标准的 Ground Truth 人物
    gt_kpts = [get_gt_keypoints(scene, 1), get_gt_keypoints(scene, 2)]
    n_gt = len(gt_kpts)

    if len(results) == 0 or results[0].keypoints is None:
        return 0.0

    h, w = img.shape[:2]
    det_kpts = results[0].keypoints.data.cpu().numpy()  # (num_people, 17, 3)

    # 转换为自定义 11 点归一化结构
    persons = []
    for kp_data in det_kpts:
        kp_17 = kp_data[:, :2] / np.array([w, h])  # 坐标归一化
        conf = kp_data[:, 2]
        kp_11 = np.full((11, 2), -1.0, dtype=np.float32)
        
        # 填充直接映射点
        for cus_idx, coco_idx in CUSTOM_TO_COCO.items():
            if conf[coco_idx] > 0.5:
                kp_11[cus_idx] = kp_17[coco_idx]
        
        # 优化点 1：通过双肩中点动态计算真实的颈部 (Index 7)
        if conf[5] > 0.5 and conf[6] > 0.5:
            kp_11[7] = (kp_17[5] + kp_17[6]) / 2.0
            
        # 优化点 2：通过双髋中点动态计算真实的骨盆 (Index 8)
        if conf[11] > 0.5 and conf[12] > 0.5:
            kp_11[8] = (kp_17[11] + kp_17[12]) / 2.0

        persons.append(kp_11)

    n_det = len(persons)
    if n_det == 0:
        return 0.0

    # 匈牙利算法进行跨角色最佳匹配
    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = -compute_oks(gt_kpts[i], persons[j], sigma=0.1)

    row_ind, col_ind = linear_sum_assignment(cost)
    total_matched_oks = sum([-cost[r, c] for r, c in zip(row_ind, col_ind)])
    
    # 优化点 3：除以 n_gt(2) 而非匹配数，未检测到或少生成人会受到科学合理的零分惩罚
    return float(total_matched_oks / n_gt)

# ===================== 背景与深度评估 =====================
def get_background_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask == 0).astype(np.uint8)

def compute_background_consistency(img1, img2, mask1, mask2):
    """ 优化点 4：精准的区域 Masked SSIM，无填充伪影，数值不虚高 """
    common_bg = (mask1 & mask2).astype(np.uint8)
    if np.sum(common_bg) == 0:
        return 1.0   # 无共同背景区域视为完全一致
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # full=True 获得全图密集的局部相似度矩阵，然后只过滤并平均真正的背景像素
    _, ssim_img = ssim(img1_gray, img2_gray, full=True, data_range=255)
    return float(np.mean(ssim_img[common_bg > 0]))

def compute_depth_fidelity(img, gt_depth_path):
    """ 深度图保真度评估 """
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        return 0.0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    
    # 优化点 5：解决通道不一致问题，将 BGR 转换为深度网络期待的 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        gen_depth = depth_model.infer_image(img_rgb, input_size=IMAGE_SIZE)
        
    # 线性归一化到 [0, 1]
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    return float(ssim(gen_depth, gt_depth, data_range=1.0))

# ===================== 主函数 =====================
def main():
    print("\n【叙事一致性定量评估系统 - 论文标准升级版】")
    print("激活指标：跨场景纯背景一致性(区域SSIM)、空间布局控制力(深度图SSIM)、多角色姿态保真度(惩罚型OKS)")

    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"错误：找不到配置文件 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # 提取所有角色+背景组合
    sequence_groups = set()
    for cfg in configs:
        sequence_groups.add((cfg["characters"][0], cfg["characters"][1], cfg["background"]))
    sequence_groups = list(sequence_groups)

    report = {}
    for method, suffix in zip(METHODS, SUFFIXES):
        bg_scores = []
        depth_scores = []
        pose_scores = []

        for char1, char2, bg_name in sequence_groups:
            seq_imgs = []
            seq_masks = []

            for scene in SCENES:
                sample_id = f"{scene}_{bg_name}_{char1}_vs_{char2}"
                img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
                if not os.path.exists(img_path):
                    continue

                mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
                gt_depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")

                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                seq_imgs.append(img)
                seq_masks.append(get_background_mask(mask_path))

                # 深度保真度
                depth_scores.append(compute_depth_fidelity(img, gt_depth_path))
                # 姿态保真度 (OKS)
                pose_scores.append(compute_pose_fidelity(img_path, scene))

            if len(seq_imgs) > 1:
                for i in range(len(seq_imgs) - 1):
                    bg_scores.append(compute_background_consistency(
                        seq_imgs[i], seq_imgs[i+1], seq_masks[i], seq_masks[i+1]
                    ))

        if not depth_scores:
            print(f">> 评估方法 {method.upper()}: 未找到任何有效图像，跳过。")
            continue

        print(f"\n>> 正在处理评估方法: {method.upper()}")
        final_bg = round(np.mean(bg_scores), 4) if bg_scores else 0.0
        final_depth = round(np.mean(depth_scores), 4) if depth_scores else 0.0
        final_pose = round(np.mean(pose_scores), 4) if pose_scores else 0.0
        final_total = round((final_bg + final_depth + final_pose) / 3, 4)

        print(f"    背景一致性 (跨场景稳定度): {final_bg:.4f}")
        print(f"    深度图保真度 (Layout控制力): {final_depth:.4f}")
        print(f"    姿态保真度 (惩罚型OKS): {final_pose:.4f}")
        print(f"    核心叙事一致性总分: {final_total:.4f}")

        report[method] = {
            "background_consistency": final_bg,
            "depth_fidelity": final_depth,
            "pose_fidelity_oks": final_pose,
            "overall_narrative_consistency": final_total,
            "total_evaluated_images": len(depth_scores)
        }

    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n评估完成！结果保存至 {SAVE_REPORT}")

if __name__ == "__main__":
    main()