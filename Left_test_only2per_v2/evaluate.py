import os
import json
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ===================== 全局配置（对齐提案实验设计） =====================
# 数据集与结果路径
SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"

# 场景与背景（与数据集生成脚本完全对齐）
SCENES = ["side_by_side", "handshake", "front_back"]
BACKGROUNDS = ["outdoor_park", "indoor_room", "futuristic_street"]

# 评估方法组：3组官方基线 + 1组消融 + 1组提出方法（与生成脚本一一对应）
METHOD_GROUPS = {
    "baseline1": "Pure text SD (no spatial control)",
    "baseline2": "SD + Single OpenPose ControlNet",
    "baseline3": "SD + Dual ControlNet (Pose + Depth, global LoRA, no mask)",
    "baseline4": "SD + Dual ControlNet + Regional LoRA only (no cross-attention mask)",
    "method": "Proposed: Dual ControlNet + Cross-Attention Mask + Regional LoRA"
}
SUFFIX_MAP = {name: f"_{name}" for name in METHOD_GROUPS.keys()}

# 通用参数
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_REPORT = "unified_synthetic_evaluation_report.json"

# ===================== 一次性加载所有评估模型 =====================
print("=" * 60)
print("[1/3] Loading evaluation models...")

# 1. CLIP 模型：用于角色身份保持与特征隔离评估（对应提案 Masked CLIP 指标）
print("    Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

# 2. YOLOv8-Pose：用于姿态布局精度评估（对应提案 姿态关键点相似度）
print("    Loading YOLOv8-pose...")
pose_model = YOLO("../yolov8n-pose.pt").to(DEVICE)

# 3. Depth Anything V2：用于深度布局精度评估（对应提案 深度一致性）
print("    Loading Depth Anything V2 (vitb)...")
depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=DEVICE), strict=True)
depth_model = depth_model.to(DEVICE).eval()

print("    All models loaded successfully.\n")

# ===================== 关键点映射与GT定义（与数据集生成脚本完全对齐） =====================
# COCO 17点 -> 自定义11点映射
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
    """
    与 generate_synthetic_dataset.py 中 get_scene_keypoints 完全一致
    保证姿态评估的基准与生成控制信号严格对齐
    """
    if scene == "side_by_side":
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.40, 0.65], [0.18, 0.78], [0.42, 0.78],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        p2 = np.array([
            [0.70, 0.42], [0.64, 0.52], [0.76, 0.52],
            [0.60, 0.65], [0.80, 0.65], [0.58, 0.78], [0.82, 0.78],
            [0.70, 0.49], [0.70, 0.72], [0.64, 0.88], [0.76, 0.88]
        ])
    elif scene == "handshake":
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
    else:  # front_back
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
    return p1 if person_idx == 1 else p2

# ===================== 工具函数1：Prompt角色描述提取 =====================
def get_person_descriptions_from_prompt(prompt, scene):
    """从标准prompt中提取两个角色的文本描述，用于CLIP文本编码"""
    scene_keyword = scene.replace('_', ' ')
    parts = prompt.split(". person2:", 1)
    
    if len(parts) == 2:
        person1_part = parts[0].replace("person1:", "", 1).strip()
        remaining = parts[1]
        scene_pattern = f". {scene_keyword}"
        if scene_pattern in remaining:
            person2_part = remaining.split(scene_pattern, 1)[0].strip()
        else:
            person2_part = remaining.split(".", 1)[0].strip()
        
        desc1 = person1_part.replace("a photo of ", "", 1).strip()
        desc2 = person2_part.replace("a photo of ", "", 1).strip()
        return desc1, desc2
    
    # 兼容回退逻辑
    prompt_lower = prompt.lower()
    if "person1:" in prompt_lower and "person2:" in prompt_lower:
        p1 = prompt_lower.split("person1:")[1].split("person2:")[0]
        p2 = prompt_lower.split("person2:")[1].split(scene_keyword)[0]
        desc1 = p1.replace("a photo of", "").strip().rstrip(',')
        desc2 = p2.replace("a photo of", "").strip().rstrip(',')
        return desc1, desc2
    
    return "character", "character"

# ===================== 工具函数2：Masked CLIP 特征隔离评估 =====================
def compute_masked_clip_metrics(img, mask, desc1, desc2):
    """
    对应提案：Feature separation and character identity preservation
    输出：自身相似度均值、交叉相似度均值、净隔离度、单角色详细相似度
    """
    text_tokens = clip.tokenize([desc1, desc2]).to(DEVICE)
    with torch.no_grad():
        text_embeds = clip_model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    sim_1_1 = sim_1_2 = sim_2_1 = sim_2_2 = 0.0
    pixel_map = {1: 128, 2: 255}
    
    for person_id, pixel_val in pixel_map.items():
        person_mask = (mask == pixel_val).astype(np.uint8)
        if np.sum(person_mask) == 0:
            continue
        
        masked_img = cv2.bitwise_and(img, img, mask=person_mask)
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        img_tensor = clip_preprocess(masked_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            img_embed = clip_model.encode_image(img_tensor)
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
    
    return {
        "masked_clip_self": diag_mean,
        "masked_clip_cross": cross_mean,
        "net_isolation": net_isolation,
        "sim_1_1": sim_1_1,
        "sim_1_2": sim_1_2,
        "sim_2_1": sim_2_1,
        "sim_2_2": sim_2_2
    }

# ===================== 工具函数3：姿态保真度 OKS 评估 =====================
def compute_oks(gt_kpts, pred_kpts, sigma=0.1):
    """计算自定义11点关键点相似度（OKS）"""
    if len(gt_kpts) != len(pred_kpts):
        return 0.0
    valid = (gt_kpts[:, 0] >= 0) & (pred_kpts[:, 0] >= 0)
    if np.sum(valid) < 3:
        return 0.0
    dist = np.linalg.norm(gt_kpts[valid] - pred_kpts[valid], axis=1)
    oks = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    return float(np.mean(oks))

def compute_pose_fidelity(img, scene):
    """
    对应提案：Posture keypoint similarity（布局精度维度）
    含匈牙利匹配 + 漏检人数惩罚
    """
    results = pose_model(img, verbose=False)
    gt_kpts = [get_gt_keypoints(scene, 1), get_gt_keypoints(scene, 2)]
    n_gt = len(gt_kpts)
    
    if len(results) == 0 or results[0].keypoints is None:
        return 0.0
    
    h, w = img.shape[:2]
    det_kpts = results[0].keypoints.data.cpu().numpy()
    persons = []
    
    for kp_data in det_kpts:
        kp_17 = kp_data[:, :2] / np.array([w, h])
        conf = kp_data[:, 2]
        kp_11 = np.full((11, 2), -1.0, dtype=np.float32)
        
        # 填充直接映射的关键点
        for cus_idx, coco_idx in CUSTOM_TO_COCO.items():
            if conf[coco_idx] > 0.5:
                kp_11[cus_idx] = kp_17[coco_idx]
        
        # 计算颈部（双肩中点）
        if conf[5] > 0.5 and conf[6] > 0.5:
            kp_11[7] = (kp_17[5] + kp_17[6]) / 2.0
        
        # 计算骨盆（双髋中点）
        if conf[11] > 0.5 and conf[12] > 0.5:
            kp_11[8] = (kp_17[11] + kp_17[12]) / 2.0
        
        persons.append(kp_11)
    
    n_det = len(persons)
    if n_det == 0:
        return 0.0
    
    # 匈牙利最优匹配
    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        for j in range(n_det):
            cost[i, j] = -compute_oks(gt_kpts[i], persons[j], sigma=0.1)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    total_matched_oks = sum([-cost[r, c] for r, c in zip(row_ind, col_ind)])
    
    # 除以GT人数，对漏检/少人生成进行惩罚
    return float(total_matched_oks / n_gt)

# ===================== 工具函数4：深度保真度评估 =====================
def compute_depth_fidelity(img, gt_depth_path):
    """
    对应提案：Depth consistency（SSIM + RMSE）
    补充提案要求的RMSE指标，同时保留SSIM
    """
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
    if gt_depth is None:
        return 0.0, 0.0
    
    gt_depth = cv2.resize(gt_depth, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad():
        gen_depth = depth_model.infer_image(img_rgb, input_size=IMAGE_SIZE)
    
    # 归一化到[0,1]
    gen_depth = (gen_depth - gen_depth.min()) / (gen_depth.max() - gen_depth.min() + 1e-8)
    gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    
    ssim_score = float(ssim(gen_depth, gt_depth, data_range=1.0))
    rmse_score = float(np.sqrt(np.mean((gen_depth - gt_depth) ** 2)))
    
    return ssim_score, rmse_score

# ===================== 工具函数5：跨场景背景一致性评估 =====================
def get_background_mask(mask):
    """从实例掩码中提取纯背景区域掩码"""
    return (mask == 0).astype(np.uint8)

def compute_background_consistency(img1, img2, mask1, mask2):
    """
    对应提案：Cross-scene narrative consistency（背景稳定性）
    采用区域Masked SSIM，仅计算公共背景区域，无填充伪影
    """
    common_bg = (mask1 & mask2).astype(np.uint8)
    if np.sum(common_bg) == 0:
        return 1.0
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    _, ssim_map = ssim(img1_gray, img2_gray, full=True, data_range=255)
    return float(np.mean(ssim_map[common_bg > 0]))

# ===================== 主评估流程 =====================
def run_unified_evaluation():
    print("=" * 60)
    print("[2/3] Starting unified synthetic dataset evaluation...")
    print("Evaluation Dimensions:")
    print("  1. Character Identity & Feature Isolation (Masked CLIP)")
    print("  2. Spatial Layout Accuracy (Pose OKS + Depth SSIM/RMSE)")
    print("  3. Cross-scene Narrative Consistency (Background SSIM)")
    print("=" * 60)
    
    # 读取数据集配置
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    # 提取唯一角色对（用于叙事一致性分组）
    unique_char_pairs = list({(cfg["characters"][0], cfg["characters"][1]) for cfg in configs})
    print(f"\n[Info] Total samples: {len(configs)}, character pairs: {len(unique_char_pairs)}")
    
    final_report = {}
    
    # 遍历每个评估方法
    for method_name, method_desc in METHOD_GROUPS.items():
        suffix = SUFFIX_MAP[method_name]
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name} ({method_desc})")
        
        # 单帧指标存储
        clip_self_list = []
        clip_cross_list = []
        net_iso_list = []
        pose_list = []
        depth_ssim_list = []
        depth_rmse_list = []
        
        # 分场景统计
        per_scene_metrics = {scene: {"clip": [], "pose": [], "depth_ssim": [], "depth_rmse": []} for scene in SCENES}
        
        # ========== 第一步：单帧全指标计算 ==========
        for cfg in tqdm(configs, desc="  Single-frame metrics"):
            sample_id = cfg["sample_id"]
            scene = cfg["scene"]
            prompt = cfg["prompt"]
            
            # 路径拼接
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
            gt_depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
            
            # 读取并预处理
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            
            # 1. 特征隔离指标（Masked CLIP）
            desc1, desc2 = get_person_descriptions_from_prompt(prompt, scene)
            clip_res = compute_masked_clip_metrics(img, mask, desc1, desc2)
            clip_self_list.append(clip_res["masked_clip_self"])
            clip_cross_list.append(clip_res["masked_clip_cross"])
            net_iso_list.append(clip_res["net_isolation"])
            per_scene_metrics[scene]["clip"].append(clip_res)
            
            # 2. 姿态保真度
            pose_score = compute_pose_fidelity(img, scene)
            pose_list.append(pose_score)
            per_scene_metrics[scene]["pose"].append(pose_score)
            
            # 3. 深度保真度
            depth_ssim, depth_rmse = compute_depth_fidelity(img, gt_depth_path)
            depth_ssim_list.append(depth_ssim)
            depth_rmse_list.append(depth_rmse)
            per_scene_metrics[scene]["depth_ssim"].append(depth_ssim)
            per_scene_metrics[scene]["depth_rmse"].append(depth_rmse)
        
        # ========== 第二步：跨场景叙事一致性计算 ==========
        bg_consistency_list = []
        for char1, char2 in unique_char_pairs:
            for bg in BACKGROUNDS:
                seq_imgs = []
                seq_bg_masks = []
                
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
                    seq_bg_masks.append(get_background_mask(mask))
                
                # 计算相邻场景的背景一致性
                if len(seq_imgs) >= 2:
                    for i in range(len(seq_imgs) - 1):
                        bg_score = compute_background_consistency(
                            seq_imgs[i], seq_imgs[i+1],
                            seq_bg_masks[i], seq_bg_masks[i+1]
                        )
                        bg_consistency_list.append(bg_score)
        
        # ========== 第三步：指标汇总 ==========
        # 全局均值
        avg_clip_self = round(float(np.mean(clip_self_list)), 4) if clip_self_list else 0.0
        avg_clip_cross = round(float(np.mean(clip_cross_list)), 4) if clip_cross_list else 0.0
        avg_net_iso = round(float(np.mean(net_iso_list)), 4) if net_iso_list else 0.0
        avg_pose = round(float(np.mean(pose_list)), 4) if pose_list else 0.0
        avg_depth_ssim = round(float(np.mean(depth_ssim_list)), 4) if depth_ssim_list else 0.0
        avg_depth_rmse = round(float(np.mean(depth_rmse_list)), 4) if depth_rmse_list else 0.0
        avg_bg_consis = round(float(np.mean(bg_consistency_list)), 4) if bg_consistency_list else 0.0
        
        # 复合得分
        layout_accuracy = round((avg_pose + avg_depth_ssim) / 2, 4)
        identity_preservation = round(avg_net_iso, 4)
        narrative_consistency = round((avg_bg_consis + avg_depth_ssim) / 2, 4)
        overall_score = round((identity_preservation + layout_accuracy + narrative_consistency) / 3, 4)
        
        # 分场景均值
        per_scene_avg = {}
        for scene in SCENES:
            data = per_scene_metrics[scene]
            per_scene_avg[scene] = {
                "masked_clip_self": round(float(np.mean([x["masked_clip_self"] for x in data["clip"]])), 4) if data["clip"] else 0.0,
                "net_isolation": round(float(np.mean([x["net_isolation"] for x in data["clip"]])), 4) if data["clip"] else 0.0,
                "pose_fidelity_oks": round(float(np.mean(data["pose"])), 4) if data["pose"] else 0.0,
                "depth_ssim": round(float(np.mean(data["depth_ssim"])), 4) if data["depth_ssim"] else 0.0,
                "depth_rmse": round(float(np.mean(data["depth_rmse"])), 4) if data["depth_rmse"] else 0.0,
                "sample_count": len(data["pose"])
            }
        
        # 存入报告
        final_report[method_name] = {
            "description": method_desc,
            "identity_preservation": {
                "masked_clip_self_similarity": avg_clip_self,
                "masked_clip_cross_similarity": avg_clip_cross,
                "net_feature_isolation": avg_net_iso
            },
            "layout_accuracy": {
                "pose_fidelity_oks": avg_pose,
                "depth_ssim": avg_depth_ssim,
                "depth_rmse": avg_depth_rmse,
                "composite_layout_score": layout_accuracy
            },
            "narrative_consistency": {
                "cross_scene_background_ssim": avg_bg_consis,
                "depth_stability": avg_depth_ssim,
                "composite_narrative_score": narrative_consistency
            },
            "per_scene_breakdown": per_scene_avg,
            "overall_score": overall_score,
            "total_evaluated_samples": len(pose_list)
        }
        
        # 控制台打印
        print(f"\n  [Identity Preservation]")
        print(f"    Self CLIP: {avg_clip_self:.4f} | Cross CLIP: {avg_clip_cross:.4f} | Net Isolation: {avg_net_iso:.4f}")
        print(f"  [Layout Accuracy]")
        print(f"    Pose OKS: {avg_pose:.4f} | Depth SSIM: {avg_depth_ssim:.4f} | Depth RMSE: {avg_depth_rmse:.4f}")
        print(f"  [Narrative Consistency]")
        print(f"    Background SSIM: {avg_bg_consis:.4f} | Layout Score: {layout_accuracy:.4f}")
        print(f"  [Overall Score]: {overall_score:.4f}")
    
    # ========== 保存报告 ==========
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"[3/3] Evaluation completed! Full report saved to {SAVE_REPORT}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_unified_evaluation()
    except Exception as e:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        raise