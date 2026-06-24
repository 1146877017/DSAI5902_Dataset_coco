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
METHODS = ["baseline1", "baseline2", "baseline3", "baseline4", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_baseline4", "_method"]
SAVE_REPORT = "masked_clip_identity_report.json"
IMAGE_SIZE = 512
BG_GRAY = 128  # 中性灰背景，减少掩码边缘对CLIP特征的干扰

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model (ViT-B/32)...")
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP model loaded successfully, device: {device}")


def extract_character_descriptions(prompt):
    """
    适配当前prompt格式: person1: {tags}. person2: {tags}. front view, {scene} scene in a {bg}
    精准提取两个人物的纯特征描述文本
    """
    print(f"  [FUNC] extract_character_descriptions called")
    print(f"  [FUNC] input prompt length: {len(prompt)} chars")
    desc1, desc2 = "character", "character"
    
    # 提取person1部分
    if "person1:" in prompt and "person2:" in prompt:
        print("  [FUNC] person1 & person2 markers found")
        p1_full = prompt.split("person1:")[1].split(". person2:")[0].strip()
        p2_full = prompt.split("person2:")[1].split(". front view")[0].strip()
        # 去除可能残留的前缀和标点
        desc1 = p1_full.strip().rstrip('.').strip()
        desc2 = p2_full.strip().rstrip('.').strip()
        print(f"  [FUNC] extracted desc1: {desc1}")
        print(f"  [FUNC] extracted desc2: {desc2}")
    else:
        print("  [WARN] person markers not found, using fallback 'character'")
    
    return desc1, desc2


def evaluate_single_sample(img_path, mask_path, prompt):
    """
    单样本身份隔离评估
    返回: 自相似度、交叉相似度、净隔离度、身份匹配正确性、全局图文相似度
    """
    print(f"\n  [EVAL] Start evaluating single sample")
    print(f"  [EVAL] image path: {img_path}")
    print(f"  [EVAL] mask path: {mask_path}")
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  [ERROR] Failed to load image or mask")
        return None
    
    print(f"  [EVAL] Image loaded successfully, shape: {img.shape}")
    print(f"  [EVAL] Mask loaded successfully, shape: {mask.shape}")
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    print(f"  [EVAL] Resized image & mask to {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # 提取人物描述并编码文本
    desc1, desc2 = extract_character_descriptions(prompt)
    text_tokens = clip.tokenize([desc1, desc2]).to(device)
    print(f"  [EVAL] Text tokenized, shape: {text_tokens.shape}")
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    print(f"  [EVAL] Text embeddings encoded & normalized, shape: {text_embeds.shape}")
    
    sim_1_1 = sim_1_2 = sim_2_1 = sim_2_2 = 0.0
    id_pixel_map = {1: 128, 2: 255}
    
    for person_idx, pixel_val in id_pixel_map.items():
        print(f"\n  [EVAL] Processing Person {person_idx} (mask pixel value = {pixel_val})")
        person_mask = (mask == pixel_val).astype(np.uint8)
        mask_pixel_count = np.sum(person_mask)
        print(f"  [EVAL] Person {person_idx} valid mask pixels: {mask_pixel_count}")
        
        if mask_pixel_count == 0:
            print(f"  [WARN] {os.path.basename(img_path)} Person{person_idx} mask empty, skipping")
            continue
        
        # 抠出人物并放置在中性灰背景上，避免纯黑背景干扰特征
        masked_img = np.full_like(img, BG_GRAY)
        masked_img[person_mask > 0] = img[person_mask > 0]
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        print(f"  [EVAL] Person {person_idx} masked image prepared")
        
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(device)
        print(f"  [EVAL] Person {person_idx} image tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        print(f"  [EVAL] Person {person_idx} image embedding encoded & normalized")
        
        sim_to_desc1 = torch.cosine_similarity(img_embed, text_embeds[0:1]).item()
        sim_to_desc2 = torch.cosine_similarity(img_embed, text_embeds[1:2]).item()
        print(f"  [EVAL] Person {person_idx} sim-to-desc1: {sim_to_desc1:.4f}")
        print(f"  [EVAL] Person {person_idx} sim-to-desc2: {sim_to_desc2:.4f}")
        
        if person_idx == 1:
            sim_1_1, sim_1_2 = sim_to_desc1, sim_to_desc2
        else:
            sim_2_1, sim_2_2 = sim_to_desc1, sim_to_desc2
    
    # 核心指标计算
    print("\n  [EVAL] Computing core metrics...")
    diag_mean = (sim_1_1 + sim_2_2) / 2  # 自相似度（身份保持度）
    cross_mean = (sim_1_2 + sim_2_1) / 2  # 交叉相似度（特征泄露程度）
    net_isolation = ((sim_1_1 - sim_1_2) + (sim_2_2 - sim_2_1)) / 2  # 净隔离度
    # 身份准确率：两个人物都正确匹配为1，否则0
    identity_correct = 1.0 if (sim_1_1 > sim_1_2 and sim_2_2 > sim_2_1) else 0.0
    
    print(f"  [EVAL] Self similarity (diag mean): {diag_mean:.4f}")
    print(f"  [EVAL] Cross similarity: {cross_mean:.4f}")
    print(f"  [EVAL] Net isolation: {net_isolation:.4f}")
    print(f"  [EVAL] Identity correct: {bool(identity_correct)}")
    
    # 全局图文一致性
    print("\n  [EVAL] Computing global text-image similarity...")
    full_img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    global_img_tensor = preprocess(full_img_pil).unsqueeze(0).to(device)
    full_text_token = clip.tokenize([prompt]).to(device)
    
    with torch.no_grad():
        global_img_embed = model.encode_image(global_img_tensor)
        global_img_embed = global_img_embed / global_img_embed.norm(dim=-1, keepdim=True)
        full_text_embed = model.encode_text(full_text_token)
        full_text_embed = full_text_embed / full_text_embed.norm(dim=-1, keepdim=True)
        global_scene_sim = torch.cosine_similarity(global_img_embed, full_text_embed).item()
    
    print(f"  [EVAL] Global text-image similarity: {global_scene_sim:.4f}")
    print("  [EVAL] Single sample evaluation finished")
    
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
    print("\n=== Masked CLIP Identity Preservation Evaluation ===")
    config_path = os.path.join(SYNTHETIC_DATASET, "synthetic_configs.json")
    print(f"Loading dataset config from: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    print(f"Successfully loaded {len(configs)} sample configs")
    
    report = {}
    
    for method_idx, (method, suffix) in enumerate(zip(METHODS, SUFFIXES), 1):
        print(f"\n{'='*60}")
        print(f"Evaluating method [{method_idx}/{len(METHODS)}]: {method.upper()}")
        print(f"File suffix: {suffix}")
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
            
            print(f"\n    [{cfg_idx}/{len(configs)}] Sample: {sample_id}")
            print(f"    Scene: {scene}, Sequence group: {seq_group}")
            
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATASET, "masks", f"{sample_id}.png")
            
            if not os.path.exists(img_path):
                print(f"    [SKIP] Generated image not found: {img_path}")
                skipped_count += 1
                continue
            print(f"    Found generated image: {img_path}")
            
            metrics = evaluate_single_sample(img_path, mask_path, prompt)
            if not metrics:
                print(f"    [SKIP] Evaluation returned None")
                skipped_count += 1
                continue
            
            metrics["sample_id"] = sample_id
            all_samples.append(metrics)
            print(f"    Sample evaluation done, added to statistics")
            
            # 按场景分组
            if scene not in scene_metrics:
                scene_metrics[scene] = []
                print(f"    New scene category created: {scene}")
            scene_metrics[scene].append(metrics)
            
            # 按序列组分组
            if seq_group not in sequence_metrics:
                sequence_metrics[seq_group] = []
                print(f"    New sequence group created: {seq_group}")
            sequence_metrics[seq_group].append(metrics)
        
        print(f"\n  Aggregating statistics for method: {method.upper()}")
        print(f"  Total valid samples: {len(all_samples)}, skipped: {skipped_count}")
        
        # 场景维度统计
        per_scene_avg = {}
        for scene, samples in scene_metrics.items():
            per_scene_avg[scene] = {
                "self_similarity": round(np.mean([s["self_similarity"] for s in samples]), 4),
                "cross_similarity": round(np.mean([s["cross_similarity"] for s in samples]), 4),
                "net_isolation": round(np.mean([s["net_isolation"] for s in samples]), 4),
                "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in samples]), 4),
                "sample_count": len(samples)
            }
            print(f"  Scene [{scene}] avg self_sim: {per_scene_avg[scene]['self_similarity']:.4f}, "
                  f"identity_acc: {per_scene_avg[scene]['identity_accuracy']:.4f}")
        
        # 序列组维度统计
        per_seq_avg = {}
        for seq, samples in sequence_metrics.items():
            per_seq_avg[seq] = {
                "self_similarity": round(np.mean([s["self_similarity"] for s in samples]), 4),
                "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in samples]), 4),
                "sample_count": len(samples)
            }
            print(f"  Sequence [{seq}] avg self_sim: {per_seq_avg[seq]['self_similarity']:.4f}, "
                  f"identity_acc: {per_seq_avg[seq]['identity_accuracy']:.4f}")
        
        # 全局统计
        global_avg = {
            "self_similarity": round(np.mean([s["self_similarity"] for s in all_samples]), 4),
            "cross_similarity": round(np.mean([s["cross_similarity"] for s in all_samples]), 4),
            "net_isolation": round(np.mean([s["net_isolation"] for s in all_samples]), 4),
            "identity_accuracy": round(np.mean([s["identity_accuracy"] for s in all_samples]), 4),
            "global_text_image_sim": round(np.mean([s["global_text_image_sim"] for s in all_samples]), 4),
            "total_samples": len(all_samples)
        }
        
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
    print(f"All {len(METHODS)} methods evaluated")
    print(f"Saving full report to: {SAVE_REPORT}")
    
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"Report saved successfully")
    print(f"\nEvaluation complete.")


if __name__ == "__main__":
    print("[ENTRY] Script started")
    main()
    print("[EXIT] Script finished")