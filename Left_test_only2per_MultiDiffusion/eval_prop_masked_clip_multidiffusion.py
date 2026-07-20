import os
import json
import cv2
import torch
import clip
import numpy as np
from PIL import Image

# ===================== 配置 =====================
SYNTHETIC_DATASET = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"

METHODS = ["multidiffusion"]
SUFFIXES = ["_multidiffusion"]

SAVE_REPORT = "masked_clip_identity_report.json"
IMAGE_SIZE = 512
BG_GRAY = 128  # 中性灰背景

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model (ViT-B/32)...")
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP model loaded successfully, device: {device}")


def extract_character_descriptions(prompt):
    """
    适配当前prompt格式并精准提取两个人物的纯特征描述文本
    """
    desc1, desc2 = "character", "character"
    if "person1:" in prompt and "person2:" in prompt:
        p1_full = prompt.split("person1:")[1].split(". person2:")[0].strip()
        p2_full = prompt.split("person2:")[1].split(". front view")[0].strip()
        desc1 = p1_full.strip().rstrip('.').strip()
        desc2 = p2_full.strip().rstrip('.').strip()
    return desc1, desc2


def evaluate_single_sample(img_path, mask_path, prompt):
    """
    单样本身份隔离评估核心算法
    """
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    
    desc1, desc2 = extract_character_descriptions(prompt)
    text_tokens = clip.tokenize([desc1, desc2]).to(device)
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    sim_1_1 = sim_1_2 = sim_2_1 = sim_2_2 = 0.0
    id_pixel_map = {1: 128, 2: 255}
    
    for person_idx, pixel_val in id_pixel_map.items():
        person_mask = (mask == pixel_val).astype(np.uint8)
        mask_pixel_count = np.sum(person_mask)
        
        if mask_pixel_count == 0:
            continue
        
        masked_img = np.full_like(img, BG_GRAY)
        masked_img[person_mask > 0] = img[person_mask > 0]
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        
        sim_to_desc1 = torch.cosine_similarity(img_embed, text_embeds[0:1]).item()
        sim_to_desc2 = torch.cosine_similarity(img_embed, text_embeds[1:2]).item()
        
        if person_idx == 1:
            sim_1_1, sim_1_2 = sim_to_desc1, sim_to_desc2
        else:
            sim_2_1, sim_2_2 = sim_to_desc1, sim_to_desc2
    
    diag_mean = (sim_1_1 + sim_2_2) / 2
    cross_mean = (sim_1_2 + sim_2_1) / 2
    net_isolation = ((sim_1_1 - sim_1_2) + (sim_2_2 - sim_2_1)) / 2
    identity_correct = 1.0 if (sim_1_1 > sim_1_2 and sim_2_2 > sim_2_1) else 0.0
    
    full_img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    global_img_tensor = preprocess(full_img_pil).unsqueeze(0).to(device)
    full_text_token = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        global_img_embed = model.encode_image(global_img_tensor)
        global_img_embed = global_img_embed / global_img_embed.norm(dim=-1, keepdim=True)
        full_text_embed = model.encode_text(full_text_token)
        full_text_embed = full_text_embed / full_text_embed.norm(dim=-1, keepdim=True)
        global_scene_sim = torch.cosine_similarity(global_img_embed, full_text_embed).item()
    
    return {
        "self_similarity": round(diag_mean, 4),
        "cross_similarity": round(cross_mean, 4),
        "net_isolation": round(net_isolation, 4),
        "identity_accuracy": identity_correct,
        "global_text_image_sim": round(global_scene_sim, 4),
        "sim_1_1": round(sim_1_1, 4),
        "sim_1_2": round(sim_1_2, 4),
        "sim_2_1": round(sim_2_1, 4),
        "sim_2_2": round(sim_2_2, 4)
    }


def main():
    print("\n=== Masked CLIP Identity Preservation Evaluation (MultiDiffusion Target Mode) ===")
    config_path = os.path.join(SYNTHETIC_DATASET, "synthetic_configs.json")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    print(f"Successfully loaded {len(configs)} sample configs")
    
    # 如果原结果存在，先读取它以保护 Baseline 数据
    report = {}
    if os.path.exists(SAVE_REPORT):
        try:
            with open(SAVE_REPORT, "r", encoding="utf-8") as f:
                report = json.load(f)
            print(f"[INFO] Found existing report. Loaded historical methods: {list(report.keys())}")
        except Exception as e:
            print(f"[WARNING] Failed to parse existing JSON, starting with fresh dictionary. Error: {e}")
    else:
        print("[INFO] No existing report found. Creating a new one.")
    
    for method_idx, (method, suffix) in enumerate(zip(METHODS, SUFFIXES), 1):
        print(f"\n{'='*60}")
        print(f"Targeting single method [{method_idx}/{len(METHODS)}]: {method.upper()}")
        print(f"{'='*60}")
        
        scene_metrics = {}
        sequence_metrics = {}
        all_samples = []
        skipped_count = 0
        
        for cfg_idx, cfg in enumerate(configs, 1):
            sample_id = cfg["sample_id"]
            scene = cfg["scene"]
            seq_group = cfg["sequence_group"]
            prompt = cfg["prompt"]
            
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATASET, "masks", f"{sample_id}.png")
            
            if not os.path.exists(img_path):
                skipped_count += 1
                continue
            
            metrics = evaluate_single_sample(img_path, mask_path, prompt)
            if not metrics:
                skipped_count += 1
                continue
            
            metrics["sample_id"] = sample_id
            all_samples.append(metrics)
            
            if scene not in scene_metrics:
                scene_metrics[scene] = []
            scene_metrics[scene].append(metrics)
            
            if seq_group not in sequence_metrics:
                sequence_metrics[seq_group] = []
            sequence_metrics[seq_group].append(metrics)
        
        print(f"\n  Aggregating statistics for method: {method.upper()}")
        print(f"  Total valid samples: {len(all_samples)}, skipped: {skipped_count}")
        
        if len(all_samples) == 0:
            print(f"  [ERROR] No valid generated images found for {method.upper()}! Check your paths.")
            continue

        per_scene_avg = {}
        for scene, samples in scene_metrics.items():
            if not samples: continue
            per_scene_avg[scene] = {
                "self_similarity": round(np.mean([s["self_similarity"] for s in samples]), 4),
                "cross_similarity": round(np.mean([s["cross_similarity"] for s in samples]), 4),
                "net_isolation": round(np.mean([s["net_isolation"] for s in samples]), 4),
                "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in samples]), 4),
                "sample_count": len(samples)
            }
        
        per_seq_avg = {}
        for seq, samples in sequence_metrics.items():
            if not samples: continue
            per_seq_avg[seq] = {
                "self_similarity": round(np.mean([s["self_similarity"] for s in samples]), 4),
                "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in samples]), 4),
                "sample_count": len(samples)
            }
        
        global_avg = {
            "self_similarity": round(np.mean([s["self_similarity"] for s in all_samples]), 4),
            "cross_similarity": round(np.mean([s["cross_similarity"] for s in all_samples]), 4),
            "net_isolation": round(np.mean([s["net_isolation"] for s in all_samples]), 4),
            "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in all_samples]), 4),
            "global_text_image_sim": round(np.mean([s["global_text_image_sim"] for s in all_samples]), 4),
            "total_samples": len(all_samples)
        }
        
        # 将新跑出的数据无缝插入或覆盖原有 report 字典中的对应项
        report[method] = {
            "per_scene_averages": per_scene_avg,
            "per_sequence_averages": per_seq_avg,
            "global_averages": global_avg
        }
        
        print(f"\n  ---- {method.upper()} Global Results ----")
        print(f"  Global self-similarity: {global_avg['self_similarity']:.4f}")
        print(f"  Global cross-similarity: {global_avg['cross_similarity']:.4f}")
        print(f"  Global net isolation: {global_avg['net_isolation']:.4f}")
        print(f"  Identity accuracy: {global_avg['identity_accuracy']:.4f}")
        print(f"  Global text-image sim: {global_avg['global_text_image_sim']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Saving merged report to: {SAVE_REPORT}")
    
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"Report updated successfully. Current methods inside: {list(report.keys())}")


if __name__ == "__main__":
    main()