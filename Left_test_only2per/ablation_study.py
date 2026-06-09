import os
import json
import numpy as np
import cv2
import torch
import clip
import re
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ===================== 配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
SCENES = ["side_by_side", "handshake", "front_back"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "ablation_study_report_enhanced.json"
IMAGE_SIZE = 512

ABLATION_GROUPS = {
    "baseline1": "纯文本 (无控制)",
    "baseline2": "仅 OpenPose 控制",
    "baseline3": "双 ControlNet + 无掩码",
    "ablation1": "双 ControlNet + 随机掩码",
    "ablation2": "仅 OpenPose + 实例掩码",
    "method": "双 ControlNet + 实例掩码"
}
SUFFIX_MAP = {k: f"_{k}" for k in ABLATION_GROUPS.keys()}

# ===================== 加载模型 =====================
print(f"[*] Loading CLIP ViT-B/32 on {DEVICE}...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

print("[*] Loading YOLOv8-pose for pose fidelity...")
pose_model = YOLO("../yolov8n-pose.pt").to(DEVICE)

print("[*] Loading Depth Anything V2 (vitb) for depth fidelity...")
depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=DEVICE), strict=True)
depth_model = depth_model.to(DEVICE).eval()

# ===================== 辅助函数 =====================
def get_person_descriptions_from_prompt(prompt, scene):
    """使用正则表达式提取 person1 和 person2 的描述"""
    scene_keyword = scene.replace('_', ' ')
    pattern = r"person1:\s*(.*?),\s*person2:\s*(.*?),\s*" + re.escape(scene_keyword)
    match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
    if match:
        desc1 = match.group(1).strip().rstrip(',')
        desc2 = match.group(2).strip().rstrip(',')
        return desc1, desc2
    else:
        # 回退
        prompt_lower = prompt.lower()
        parts_p1 = prompt_lower.split("person1:")
        if len(parts_p1) < 2:
            return "character", "character"
        parts_p2 = parts_p1[1].split("person2:")
        if len(parts_p2) < 2:
            return "character", "character"
        desc1 = parts_p2[0].strip().rstrip(',')
        desc2 = parts_p2[1].split(scene_keyword)[0].strip().rstrip(',')
        return desc1, desc2

def compute_clip_scores(img, mask, desc1, desc2):
    """返回对角相似度、交叉相似度、净隔离得分"""
    # 文本编码
    text_tokens = clip.tokenize([desc1, desc2]).to(DEVICE)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    sim_1_1, sim_1_2, sim_2_1, sim_2_2 = 0.0, 0.0, 0.0, 0.0
    for person_id in [1, 2]:
        person_mask = (mask == person_id).astype(np.uint8)
        if np.sum(person_mask) == 0:
            continue
        masked_img = cv2.bitwise_and(img, img, mask=person_mask)
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        sim1 = torch.cosine_similarity(img_embed, text_embeds[0:1]).item()
        sim2 = torch.cosine_similarity(img_embed, text_embeds[1:2]).item()
        if person_id == 1:
            sim_1_1, sim_1_2 = sim1, sim2
        else:
            sim_2_1, sim_2_2 = sim1, sim2

    diag_mean = (sim_1_1 + sim_2_2) / 2
    cross_mean = (sim_1_2 + sim_2_1) / 2
    net_isolation = ((sim_1_1 - sim_1_2) + (sim_2_2 - sim_2_1)) / 2
    return diag_mean, cross_mean, net_isolation

def compute_background_consistency(img1, img2, mask1, mask2):
    common_bg = (mask1 == 0) & (mask2 == 0)
    if np.sum(common_bg) == 0:
        return 0.0
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_filled = img1_gray.copy().astype(np.float64)
    img2_filled = img2_gray.copy().astype(np.float64)
    bg_val1 = np.mean(img1_gray[common_bg])
    bg_val2 = np.mean(img2_gray[common_bg])
    img1_filled[~common_bg] = bg_val1
    img2_filled[~common_bg] = bg_val2
    # 转回 uint8
    img1_filled = np.clip(img1_filled, 0, 255).astype(np.uint8)
    img2_filled = np.clip(img2_filled, 0, 255).astype(np.uint8)
    return ssim(img1_filled, img2_filled, data_range=255)

def compute_depth_fidelity(img, gt_depth_path):
    """生成图像深度图与 GT 深度图的 SSIM"""
    gen_depth = depth_model.infer_image(img, input_size=IMAGE_SIZE)
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        return 0.0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    return ssim(gen_depth, gt_depth, data_range=1.0)

def compute_pose_fidelity(img, gt_pose_path):
    """生成图像与 GT 姿态图的结构相似度（灰度 SSIM）"""
    gen_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gt_pose = cv2.imread(gt_pose_path, cv2.IMREAD_GRAYSCALE)
    if gt_pose is None:
        return 0.0
    gt_pose = cv2.resize(gt_pose, (IMAGE_SIZE, IMAGE_SIZE))
    # 数据类型为 uint8
    gen_gray = gen_gray.astype(np.uint8)
    gt_pose = gt_pose.astype(np.uint8)
    return ssim(gen_gray, gt_pose, data_range=255)


# ===================== 主流程 =====================
def run_ablation():
    print("\n[>>>] 开始消融实验量化评估...")
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"[-] 错误: 找不到 {config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # 提取所有唯一角色对
    unique_pairs = set()
    for cfg in configs:
        unique_pairs.add((cfg["characters"][0], cfg["characters"][1]))
    unique_pairs = list(unique_pairs)
    print(f"[*] 检测到 {len(unique_pairs)} 组角色配对")

    report = {}
    for group_name, group_desc in ABLATION_GROUPS.items():
        suffix = SUFFIX_MAP[group_name]
        print(f"\n[*] 评估组: {group_name} ({group_desc})")

        # 存储各指标
        clip_self_scores = []
        clip_cross_scores = []
        net_isolation_scores = []
        bg_scores = []
        depth_scores = []
        pose_scores = []

        # 遍历每个样本
        for cfg in tqdm(configs, desc=f"  处理 {group_name}"):
            sample_id = cfg["sample_id"]
            scene = cfg["scene"]
            prompt = cfg["prompt"]
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
            gt_depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")
            gt_pose_path = os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            # CLIP 相关指标
            desc1, desc2 = get_person_descriptions_from_prompt(prompt, scene)
            diag, cross, net_iso = compute_clip_scores(img, mask, desc1, desc2)
            clip_self_scores.append(diag)
            clip_cross_scores.append(cross)
            net_isolation_scores.append(net_iso)

            # 深度和姿态保真度
            depth_fid = compute_depth_fidelity(img, gt_depth_path)
            pose_fid = compute_pose_fidelity(img, gt_pose_path)
            depth_scores.append(depth_fid)
            pose_scores.append(pose_fid)

        # 背景一致性（跨场景相邻帧之间的共同背景区域 SSIM）
        # 这里按角色对聚合三个场景，然后计算相邻帧的背景一致性
        for char1, char2 in unique_pairs:
            seq_imgs = []
            seq_masks = []
            for scene in SCENES:
                sample_id = f"{scene}_{char1}_vs_{char2}"
                img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
                mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
                if not os.path.exists(img_path):
                    break
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    break
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                seq_imgs.append(img)
                seq_masks.append(mask)
            if len(seq_imgs) != len(SCENES):
                continue
            for i in range(len(seq_imgs)-1):
                bg = compute_background_consistency(seq_imgs[i], seq_imgs[i+1], seq_masks[i], seq_masks[i+1])
                bg_scores.append(bg)

        # 汇总平均
        avg_clip_self = round(float(np.mean(clip_self_scores)), 4) if clip_self_scores else 0.0
        avg_clip_cross = round(float(np.mean(clip_cross_scores)), 4) if clip_cross_scores else 0.0
        avg_net_iso = round(float(np.mean(net_isolation_scores)), 4) if net_isolation_scores else 0.0
        avg_bg = round(float(np.mean(bg_scores)), 4) if bg_scores else 0.0
        avg_depth = round(float(np.mean(depth_scores)), 4) if depth_scores else 0.0
        avg_pose = round(float(np.mean(pose_scores)), 4) if pose_scores else 0.0

        # 布局总体得分（深度+姿态的平均）
        layout_score = round((avg_depth + avg_pose) / 2, 4)
        # 综合得分：净隔离 + 背景一致性 + 布局总分 的平均
        overall = round((avg_net_iso + avg_bg + layout_score) / 3, 4)

        report[group_name] = {
            "description": group_desc,
            "masked_clip_self": avg_clip_self,
            "masked_clip_cross": avg_clip_cross,
            "net_isolation": avg_net_iso,
            "background_consistency": avg_bg,
            "depth_fidelity": avg_depth,
            "pose_fidelity": avg_pose,
            "layout_accuracy_score": layout_score,
            "overall_score": overall
        }

        print(f"    -> Self CLIP: {avg_clip_self:.4f} | Cross CLIP: {avg_clip_cross:.4f} | NetISO: {avg_net_iso:.4f}")
        print(f"    -> Background: {avg_bg:.4f} | Depth: {avg_depth:.4f} | Pose: {avg_pose:.4f}")
        print(f"    -> Layout Score: {layout_score:.4f} | Overall: {overall:.4f}\n")

    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"消融实验完成，结果保存至 {SAVE_REPORT}")

if __name__ == "__main__":
    run_ablation()