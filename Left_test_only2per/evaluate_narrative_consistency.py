import os
import json
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

# ===================== 配置=====================
# 合成测试集
SYNTHETIC_DATA = "synthetic_test_dataset"
# 生成结果
GEN_RESULTS = "synthetic_results"
# 叙事序列：同一故事的连续场景
NARRATIVE_SEQUENCE = ["side_by_side", "handshake", "front_back"]
# 评估方法
METHODS = ["baseline1", "baseline2", "baseline3", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_method"]

IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "narrative_consistency_report.json"

# ===================== 评估模型 =====================
print(" 加载姿态模型 + 深度模型...")
pose_model = YOLO("yolov8n-pose.pt")  # 人物位置
# 深度模型
from depth_anything.dpt import DPT_DINOv2
depth_model = DPT_DINOv2(encoder='dinov2_large', features=256, out_channels=[256, 512, 1024, 1024])
depth_model.load_state_dict(torch.load("depth_anything_v2_vitl.pth", map_location=DEVICE), strict=False)
depth_model = depth_model.to(DEVICE).eval()

# ===================== 评估函数 =====================

def get_background_mask(mask_path):
    """获取背景掩码"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    bg_mask = (mask == 0).astype(np.float32)  # 0=背景
    return bg_mask

def compute_background_consistency(img1, img2, bg_mask):
    """计算两张图背景相似度SSIM"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 计算背景区域
    img1_bg = img1_gray * bg_mask
    img2_bg = img2_gray * bg_mask
    return ssim(img1_bg, img2_bg, data_range=255)

def compute_depth_consistency(depth1, depth2):
    """深度图一致性SSIM"""
    return ssim(depth1, depth2, data_range=1.0)

def compute_pose_relative_consistency(kps1, kps2):
    """人物相对位置一致性：
    两个人物之间的距离比例保持不变--叙事空间稳定
    """
    if len(kps1) < 2 or len(kps2) < 2:
        return 0.0
    # 取中心点
    c1 = np.mean(kps1[0][:, :2], axis=0)
    c2 = np.mean(kps1[1][:, :2], axis=0)
    dist1 = np.linalg.norm(c1 - c2)

    c3 = np.mean(kps2[0][:, :2], axis=0)
    c4 = np.mean(kps2[1][:, :2], axis=0)
    dist2 = np.linalg.norm(c3 - c4)

    # 比例一致性
    ratio = min(dist1, dist2) / max(dist1, dist2)
    return ratio

def get_depth_map(img):
    """获取深度图"""
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = depth_model(img)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear'
        ).squeeze().cpu().numpy()
    # 归一化
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth

def get_keypoints(img):
    """获取两个人物关键点"""
    res = pose_model(img, verbose=False)
    if len(res[0].keypoints.xy) >= 2:
        return res[0].keypoints.xy.cpu().numpy()[:2]
    return []

# ===================== 评估流程 =====================
def main():
    print("\n 开始运行叙事一致性评估")
    print(" 评估指标：背景一致性 + 深度一致性 + 人物相对位置一致性\n")

    report = {}

    # 遍历每个方法
    for method, suffix in zip(METHODS, SUFFIXES):
        print(f" 评估方法：{method.upper()}")

        # 加载序列内所有生成图
        seq_imgs = []
        seq_masks = []
        seq_depths = []
        seq_kps = []

        for scene in NARRATIVE_SEQUENCE:
            # 读取图像
            img_path = os.path.join(GEN_RESULTS, f"{scene}{suffix}.png")
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            seq_imgs.append(img)

            # 掩码
            mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{scene}.png")
            seq_masks.append(get_background_mask(mask_path))

            # 深度
            seq_depths.append(get_depth_map(img))

            # 关键点
            seq_kps.append(get_keypoints(img))

        # 计算序列一致性 : 相邻帧对比
        bg_scores = []
        depth_scores = []
        pose_scores = []

        for i in range(len(seq_imgs) - 1):
            img_a, mask_a = seq_imgs[i], seq_masks[i]
            img_b, mask_b = seq_imgs[i+1], seq_masks[i+1]
            depth_a, depth_b = seq_depths[i], seq_depths[i+1]
            kps_a, kps_b = seq_kps[i], seq_kps[i+1]

            bg = compute_background_consistency(img_a, img_b, mask_a)
            dp = compute_depth_consistency(depth_a, depth_b)
            ps = compute_pose_relative_consistency(kps_a, kps_b)

            bg_scores.append(bg)
            depth_scores.append(dp)
            pose_scores.append(ps)

        # 最终一致性得分
        final_bg = round(float(np.mean(bg_scores)), 4)
        final_depth = round(float(np.mean(depth_scores)), 4)
        final_pose = round(float(np.mean(pose_scores)), 4)
        final_total = round((final_bg + final_depth + final_pose) / 3, 4)

        report[method] = {
            "background_consistency": final_bg,
            "depth_consistency": final_depth,
            "pose_relative_consistency": final_pose,
            "overall_narrative_consistency": final_total
        }

        print(f"   背景一致性: {final_bg:.4f}")
        print(f"   深度一致性: {final_depth:.4f}")
        print(f"   位置一致性: {final_pose:.4f}")
        print(f"   总叙事一致性: {final_total:.4f}\n")

    # 保存
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(" 叙事一致性评估完成！")
    print(f" 保存：{SAVE_REPORT}")

if __name__ == "__main__":
    main()