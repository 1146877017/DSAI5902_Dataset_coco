import os
import json
import numpy as np
import cv2
import torch
import clip
import re
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ===================== 配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
SCENES = ["side_by_side", "handshake", "front_back"]
BACKGROUNDS = ["outdoor_park", "indoor_room", "futuristic_street"]  # 用于对齐标准数据集路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "ablation_study_report_enhanced.json"
IMAGE_SIZE = 512

ABLATION_GROUPS = {
    "baseline1": "Pure text (no control)",
    "baseline2": "OpenPose only",
    "baseline3": "Dual ControlNet + no mask",
    "method": "Dual ControlNet + instance mask"
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
    """
    Extract descriptions of person1 and person2 from prompt.
    Actual format: person1: a photo of {tags}. person2: a photo of {tags}. {scene} scene in a {bg}, ...
    Returns (desc1, desc2) with "a photo of " prefix removed.
    """
    scene_keyword = scene.replace('_', ' ')
    # split by ". person2:" to get person1 part and the rest
    parts = prompt.split(". person2:", 1)
    if len(parts) == 2:
        # person1 part: "person1: a photo of ..."
        person1_part = parts[0].replace("person1:", "", 1).strip()
        # remaining: " a photo of {tags}. {scene} scene ..."
        remaining = parts[1]
        # split by f". {scene_keyword}" to get person2 description
        scene_pattern = f". {scene_keyword}"
        if scene_pattern in remaining:
            person2_part = remaining.split(scene_pattern, 1)[0].strip()
        else:
            # fallback: take up to first period
            person2_part = remaining.split(".", 1)[0].strip()
        # remove "a photo of " prefix
        desc1 = person1_part.replace("a photo of ", "", 1).strip()
        desc2 = person2_part.replace("a photo of ", "", 1).strip()
        return desc1, desc2

    # second fallback: simple split
    prompt_lower = prompt.lower()
    if "person1:" in prompt_lower and "person2:" in prompt_lower:
        p1 = prompt_lower.split("person1:")[1].split("person2:")[0]
        p2 = prompt_lower.split("person2:")[1].split(scene_keyword)[0]
        desc1 = p1.replace("a photo of", "").strip().rstrip(',')
        desc2 = p2.replace("a photo of", "").strip().rstrip(',')
        return desc1, desc2
    return "character", "character"

def compute_clip_scores(img, mask, desc1, desc2):
    """Return diagonal similarity, cross similarity, net isolation score"""
    text_tokens = clip.tokenize([desc1, desc2]).to(DEVICE)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    sim_1_1, sim_1_2, sim_2_1, sim_2_2 = 0.0, 0.0, 0.0, 0.0
    
    pixel_map = {1: 128, 2: 255}
    for person_id, pixel_val in pixel_map.items():
        person_mask = (mask == pixel_val).astype(np.uint8)
        if np.sum(person_mask) == 0:
            continue
        masked_img = cv2.bitwise_and(img, img, mask=person_mask)
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        # sim1 = torch.cosine_similarity(img_embed, text_embeds[0:1]).item()
        # sim2 = torch.cosine_similarity(img_embed, text_embeds[1:2]).item()
        sim1 = torch.matmul(img_embed, text_embeds[0:1].T).item()
        sim2 = torch.matmul(img_embed, text_embeds[1:2].T).item()
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
        return 1.0
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_filled = img1_gray.copy().astype(np.float64)
    img2_filled = img2_gray.copy().astype(np.float64)
    bg_val1 = np.mean(img1_gray[common_bg])
    bg_val2 = np.mean(img2_gray[common_bg])
    img1_filled[~common_bg] = bg_val1
    img2_filled[~common_bg] = bg_val2
    img1_filled = np.clip(img1_filled, 0, 255).astype(np.uint8)
    img2_filled = np.clip(img2_filled, 0, 255).astype(np.uint8)
    return ssim(img1_filled, img2_filled, data_range=255)

def compute_depth_fidelity(img, gt_depth_path):
    """SSIM between generated depth map and GT depth map"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gen_depth = depth_model.infer_image(img_rgb, input_size=IMAGE_SIZE)
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        return 0.0
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE))
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    return ssim(gen_depth, gt_depth, data_range=1.0)

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
    """Return standard keypoints of the scene (consistent with get_scene_keypoints in generate_synthetic_dataset.py)"""
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
    """Compute OKS for custom 11-point keypoints"""
    if len(gt_kpts) != len(pred_kpts):
        return 0.0
    valid = (gt_kpts[:,0] >= 0) & (pred_kpts[:,0] >= 0)
    if np.sum(valid) < 3:
        return 0.0
    dist = np.linalg.norm(gt_kpts[valid] - pred_kpts[valid], axis=1)
    oks = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    return float(np.mean(oks))

def compute_pose_fidelity(img, scene):
    """Core pose fidelity evaluation with missing person penalty"""
    
    results = pose_model(img, verbose=False)
    
    # Two ground truth persons
    gt_kpts = [get_gt_keypoints(scene, 1), get_gt_keypoints(scene, 2)]
    n_gt = len(gt_kpts)

    if len(results) == 0 or results[0].keypoints is None:
        return 0.0

    h, w = img.shape[:2]
    det_kpts = results[0].keypoints.data.cpu().numpy()  # (num_people, 17, 3)

    # Convert to custom 11-point normalized structure
    persons = []
    for kp_data in det_kpts:
        kp_17 = kp_data[:, :2] / np.array([w, h])  # normalize coordinates
        conf = kp_data[:, 2]
        kp_11 = np.full((11, 2), -1.0, dtype=np.float32)
        
        # Fill directly mapped points
        for cus_idx, coco_idx in CUSTOM_TO_COCO.items():
            if conf[coco_idx] > 0.5:
                kp_11[cus_idx] = kp_17[coco_idx]
        
        # Compute neck (Index 7) as midpoint of shoulders
        if conf[5] > 0.5 and conf[6] > 0.5:
            kp_11[7] = (kp_17[5] + kp_17[6]) / 2.0
            
        # Compute pelvis (Index 8) as midpoint of hips
        if conf[11] > 0.5 and conf[12] > 0.5:
            kp_11[8] = (kp_17[11] + kp_17[12]) / 2.0

        persons.append(kp_11)

    n_det = len(persons)
    if n_det == 0:
        return 0.0

    # Hungarian matching for best assignment
    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = -compute_oks(gt_kpts[i], persons[j], sigma=0.1)

    row_ind, col_ind = linear_sum_assignment(cost)
    total_matched_oks = sum([-cost[r, c] for r, c in zip(row_ind, col_ind)])
    
    # Divide by n_gt (2) to penalize missing detections
    return float(total_matched_oks / n_gt)

# ===================== 主流程 =====================
def run_ablation():
    print("\n[>>>] Starting ablation study quantitative evaluation...")
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"[-] Error: {config_path} not found")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # Extract all unique character pairs
    unique_pairs = set()
    for cfg in configs:
        unique_pairs.add((cfg["characters"][0], cfg["characters"][1]))
    unique_pairs = list(unique_pairs)
    print(f"[*] Detected {len(unique_pairs)} character pairs")

    report = {}
    for group_name, group_desc in ABLATION_GROUPS.items():
        suffix = SUFFIX_MAP[group_name]
        print(f"\n[*] Evaluation Group: {group_name} ({group_desc})")

        # Storage for metrics
        clip_self_scores = []
        clip_cross_scores = []
        net_isolation_scores = []
        bg_scores = []
        depth_scores = []
        pose_scores = []

        # Single-frame metrics for each sample
        for cfg in tqdm(configs, desc=f"  Processing {group_name}"):
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

            # 1. CLIP text‑local feature matching metrics
            desc1, desc2 = get_person_descriptions_from_prompt(prompt, scene)
            diag, cross, net_iso = compute_clip_scores(img, mask, desc1, desc2)
            clip_self_scores.append(diag)
            clip_cross_scores.append(cross)
            net_isolation_scores.append(net_iso)

            # 2. Depth and pose fidelity
            depth_fid = compute_depth_fidelity(img, gt_depth_path)
            pose_fid = compute_pose_fidelity(img, scene)  #  
            depth_scores.append(depth_fid)
            pose_scores.append(pose_fid)

        # 3. Background consistency across scenes
        for char1, char2 in unique_pairs:
            for bg in BACKGROUNDS:
                seq_imgs = []
                seq_masks = []
                for scene in SCENES:
                    sample_id = f"{scene}_{bg}_{char1}_vs_{char2}"
                    img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
                    mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
                    
                    if not os.path.exists(img_path) or not os.path.exists(mask_path):
                        continue
                    img = cv2.imread(img_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if img is None or mask is None:
                        continue
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                    seq_imgs.append(img)
                    seq_masks.append(mask)
                    
                if len(seq_imgs) < 2:
                    continue
                for i in range(len(seq_imgs) - 1):
                    bg_val = compute_background_consistency(seq_imgs[i], seq_imgs[i+1], seq_masks[i], seq_masks[i+1])
                    bg_scores.append(bg_val)

        # Aggregate averages
        avg_clip_self = round(float(np.mean(clip_self_scores)), 4) if clip_self_scores else 0.0
        avg_clip_cross = round(float(np.mean(clip_cross_scores)), 4) if clip_cross_scores else 0.0
        avg_net_iso = round(float(np.mean(net_isolation_scores)), 4) if net_isolation_scores else 0.0
        avg_bg = round(float(np.mean(bg_scores)), 4) if bg_scores else 0.0
        avg_depth = round(float(np.mean(depth_scores)), 4) if depth_scores else 0.0
        avg_pose = round(float(np.mean(pose_scores)), 4) if pose_scores else 0.0

        # Layout overall score (average of depth and pose)
        layout_score = round((avg_depth + avg_pose) / 2, 4)
        # Overall: net isolation + background stability + layout accuracy
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
        print(f"    -> Background: {avg_bg:.4f} | Depth: {avg_depth:.4f} | Pose (YOLO OKS): {avg_pose:.4f}")
        print(f"    -> Layout Score: {layout_score:.4f} | Overall: {overall:.4f}\n")

    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Ablation study quantitative evaluation completed, full report successfully saved to {SAVE_REPORT}")

if __name__ == "__main__":
    run_ablation()