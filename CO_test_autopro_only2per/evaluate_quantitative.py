import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ===================== 模式与区间控制 =====================
# 若设为 True，则自动读取汇聚后的全量数据进行评估；若为 False，则按区间读取分批结果
USE_CONSOLIDATED = True  

# 分批模式下的区间参数（当 USE_CONSOLIDATED = False 时生效）
START_IDX = 1501  
END_IDX = 1647  

# ===================== 路径与环境配置 =====================
GT_BASE = r"co_dat_only2per/complete_samples_512_filter"
GT_RAW = os.path.join(GT_BASE, "raw")
IMAGE_SIZE = 512

# 动态根据模式生成清单路径及报告保存名称
if USE_CONSOLIDATED:
    MANIFEST_PATH = "eval_manifest_all.json"
    SAVE_PATH = "quant_eval_res_all.json"
    print(f" 启动定量评估流程，当前目标：[全局全量数据汇总评估]")
else:
    MANIFEST_PATH = f"eval_manifest_{START_IDX}_{END_IDX}.json" 
    SAVE_PATH = f"quantitative_evaluation_report_{START_IDX}_{END_IDX}.json"
    print(f" 启动定量评估流程，当前目标区间: {START_IDX} 至 {END_IDX}")

print(f" 正在检索实验清单: {MANIFEST_PATH}")

# ===================== 模型初始化 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO("../yolov8n-pose.pt").to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
# depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=device), strict=False)

depth_model.load_state_dict(torch.load("../depth_anything_v2_vitb.pth", map_location=device), strict=True) # 先设为 True 测试是否完美匹配
depth_model = depth_model.to(device).eval()

# ===================== 核心评估函数 =====================
def extract_pose_keypoints(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    results = pose_model(img, conf=0.3, verbose=False)
    
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None
    kps = results[0].keypoints.xy.cpu().numpy()
    if len(kps) >= 2:
        return kps[:2]
    return None

def compute_pckh(gt_kps, gen_kps, threshold=0.5):
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
    # gen_img 是由 cv2.imread 读取的原始 BGR 图像，直接传给官方接口
    with torch.no_grad():
        # infer_image 内部会自动处理 RGB 转换、ImageNet 归一化以及尺寸缩放
        depth = depth_model.infer_image(gen_img, input_size=IMAGE_SIZE)
    
    # 归一化到 [0, 1] 以便和 GT 深度图对比
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
        gen_masked = cv2.bitwise_and(gen_img, gen_img, mask=p_mask)
        
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

# ===================== 主运行程序 =====================
def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"[-] 找不到清单文件: {MANIFEST_PATH}，请确认数据是否生成或汇聚成功。")
        return
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    METHODS = ["baseline1", "baseline2", "baseline3", "method"]
    suffix_map = {
        "baseline1": "_baseline1.png",
        "baseline2": "_baseline2.png",
        "baseline3": "_baseline3.png",
        "method": "_method.png"
    }
    
    final_report = {}
    gt_kps_cache = {}  # 性能优化：用于缓存 GT 骨骼点，避免千次重复计算
    
    for method in METHODS:
        # 将指标按模式（original_image / pure_background）解耦初始化
        metrics = {
            "original_image": {
                "pose_pckh": [], "depth_rmse": [], "depth_ssim": [], "clip_text_sim": [], "layout_correct": 0
            },
            "pure_background": {
                "pose_pckh": [], "depth_rmse": [], "depth_ssim": [], "clip_text_sim": [], "layout_correct": 0
            }
        }
        
        print(f"\n[+] 开始评测方法: {method} ...")
        
        for item in manifest_data:
            sample_name = item["sample_name"]
            mode = item["mode"]  # "original_image" 或 "pure_background"
            
            # 动态定位图像文件夹
            if USE_CONSOLIDATED:
                output_dir = f"results_all_{mode}"
            else:
                output_dir = f"results_{START_IDX}to{END_IDX}_{mode}" 
                
            gen_path = os.path.join(output_dir, f"{sample_name}{suffix_map[method]}")
            gt_raw_path = os.path.join(GT_RAW, f"{sample_name}.jpg")
            gt_depth_path = os.path.join(GT_BASE, f"depth_{mode}", f"{sample_name}.png") 
            gt_mask_path = os.path.join(GT_BASE, "mask", f"{sample_name}.png")
                                
            gen_img = cv2.imread(gen_path)
            if gen_img is None: 
                print(f"[-] 警报: 缺失图像 {gen_path}，已跳过该样本。")
                continue
            
            # 1. Pose 评估（带缓存优化）
            if sample_name not in gt_kps_cache:
                gt_kps_cache[sample_name] = extract_pose_keypoints(gt_raw_path)
            gt_kps = gt_kps_cache[sample_name]
            
            gen_kps = extract_pose_keypoints(gen_path)
            pckh_val = compute_pckh(gt_kps, gen_kps)
            metrics[mode]["pose_pckh"].append(pckh_val)
            
            # 2. Depth 评估
            rmse, ssim_val = compute_depth_metrics(gen_img, gt_depth_path)
            metrics[mode]["depth_rmse"].append(rmse)
            metrics[mode]["depth_ssim"].append(ssim_val)
            
            # 3. CLIP 文本相似度评估
            clip_sim = compute_split_person_clip(gen_img, gt_mask_path, item["person1_gt"], item["person2_gt"])
            metrics[mode]["clip_text_sim"].append(clip_sim)
            
            # 4. 布局准确率自适应判定
            if mode == "original_image":
                # 原图模式：检测出合理的两组骨骼点，且深度保留完好
                if pckh_val > 0.5 and ssim_val > 0.6:
                    metrics[mode]["layout_correct"] += 1
            elif mode == "pure_background":
                # 纯背景模式：成功消去了所有人（检测不到2人），且背景深度场一致
                if gen_kps is None and ssim_val > 0.6:
                    metrics[mode]["layout_correct"] += 1
                    
        # 组装最终的分模式报告结构
        final_report[method] = {}
        for m in ["original_image", "pure_background"]:
            m_data = metrics[m]
            total = len(m_data["pose_pckh"]) if len(m_data["pose_pckh"]) > 0 else 1
            
            final_report[method][m] = {
                "Pose PCKh@0.5": round(np.mean(m_data["pose_pckh"]), 4),
                "Depth RMSE": round(np.mean(m_data["depth_rmse"]), 4),
                "Depth SSIM": round(np.mean(m_data["depth_ssim"]), 4),
                "Masked Text-Image CLIP Similarity": round(np.mean(m_data["clip_text_sim"]), 4),
                "Layout Accuracy": round(m_data["layout_correct"] / total, 4)
            }
            print(f"  -> 模式 [{m}] 评测完毕。布局准确率: {final_report[method][m]['Layout Accuracy']:.2%}")
        
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"\n[+] 定量评估全面完成！最终解耦结果报告已保存至 `{SAVE_PATH}`")

if __name__ == "__main__":
    main()