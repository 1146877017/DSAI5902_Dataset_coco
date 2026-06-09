import os
import json
import cv2
import torch
import clip
import numpy as np
import re
from PIL import Image

# ===================== 配置 =====================
SYNTHETIC_DATASET = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
SCENES = ["side_by_side", "handshake", "front_back"]
METHODS = ["baseline1", "baseline2", "baseline3", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_method"]
SAVE_REPORT = "proposal_masked_clip_report_enhanced.json"

IMAGE_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

print(" 加载 CLIP 模型...")
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== prompt 拆分 =====================
def get_person_descriptions_from_prompt(prompt, scene):
    """
    使用正则表达式提取 person1 和 person2 的描述，忽略大小写和额外空格。
    返回 (desc1, desc2)
    """
    # 转义场景关键字
    scene_keyword = scene.replace('_', ' ')
    # 构建正则：person1: ... , person2: ... , scene_keyword
    # 描述中可能包含逗号，因此匹配到 ", person2:" 或 ", {scene_keyword}" 为止
    pattern = r"person1:\s*(.*?),\s*person2:\s*(.*?),\s*" + re.escape(scene_keyword)
    match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
    if match:
        desc1 = match.group(1).strip().rstrip(',')
        desc2 = match.group(2).strip().rstrip(',')
        return desc1, desc2
    else:
        # 回退：简单分割
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

# ===================== 核心评估函数 =====================
def evaluate_sample(img_path, mask_path, prompt, scene):
    """
    返回字典，包含：
        - masked_clip_self: 对角相似度平均值
        - cross_similarity: 交叉相似度平均值 (Sim(Mask1, Text2) + Sim(Mask2, Text1))/2
        - net_isolation: (Sim1- Sim_cross1 + Sim2 - Sim_cross2)/2
        - global_scene_sim: 完整图像与完整 prompt 的 CLIP 相似度
    """
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # 提取角色描述
    desc1, desc2 = get_person_descriptions_from_prompt(prompt, scene)
    text_tokens = clip.tokenize([desc1, desc2]).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 准备存储四个相似度：对角和交叉
    sim_1_1, sim_1_2, sim_2_1, sim_2_2 = 0.0, 0.0, 0.0, 0.0

    for person_id in [1, 2]:
        person_mask = (mask == person_id).astype(np.uint8) * 255
        if np.sum(person_mask) == 0:
            continue
        masked_img = cv2.bitwise_and(img, img, mask=person_mask)
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        # 计算与两个文本的相似度
        sim1 = torch.cosine_similarity(img_embed, text_embeds[0:1]).item()
        sim2 = torch.cosine_similarity(img_embed, text_embeds[1:2]).item()
        if person_id == 1:
            sim_1_1, sim_1_2 = sim1, sim2
        else:
            sim_2_1, sim_2_2 = sim1, sim2

    # 对角平均
    diag_mean = (sim_1_1 + sim_2_2) / 2
    # 交叉平均
    cross_mean = (sim_1_2 + sim_2_1) / 2
    # 净隔离得分：(对角-交叉) 的平均
    net_isolation = ((sim_1_1 - sim_1_2) + (sim_2_2 - sim_2_1)) / 2

    # 全局场景一致性：完整图与完整 prompt
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
        "masked_clip_self": round(diag_mean, 4),
        "cross_similarity": round(cross_mean, 4),
        "net_isolation": round(net_isolation, 4),
        "global_scene_sim": round(global_scene_sim, 4),
        "sim_1_1": round(sim_1_1, 4),
        "sim_1_2": round(sim_1_2, 4),
        "sim_2_1": round(sim_2_1, 4),
        "sim_2_2": round(sim_2_2, 4)
    }

# ===================== 评估流程 =====================
def main():
    print("\n【增强版 Masked CLIP + 场景一致性评估】")
    config_path = os.path.join(SYNTHETIC_DATASET, "synthetic_configs.json")
    if not os.path.exists(config_path):
        print(f"错误：找不到 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    report = {}
    for method, suffix in zip(METHODS, SUFFIXES):
        print(f" 评估方法：{method.upper()}")
        # 按场景存储详细指标
        scene_metrics = {scene: [] for scene in SCENES}
        all_samples = []

        for cfg in configs:
            sample_id = cfg["sample_id"]
            scene = cfg["scene"]
            prompt = cfg["prompt"]
            img_path = os.path.join(GEN_RESULTS, f"{sample_id}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATASET, "masks", f"{sample_id}.png")
            if not os.path.exists(img_path):
                continue
            metrics = evaluate_sample(img_path, mask_path, prompt, scene)
            if metrics:
                metrics["sample_id"] = sample_id
                scene_metrics[scene].append(metrics)
                all_samples.append(metrics)

        # 计算每个场景的平均值
        per_scene_avg = {}
        for scene, samples in scene_metrics.items():
            if samples:
                avg = {
                    "masked_clip_self": round(np.mean([s["masked_clip_self"] for s in samples]), 4),
                    "cross_similarity": round(np.mean([s["cross_similarity"] for s in samples]), 4),
                    "net_isolation": round(np.mean([s["net_isolation"] for s in samples]), 4),
                    "global_scene_sim": round(np.mean([s["global_scene_sim"] for s in samples]), 4),
                    "count": len(samples)
                }
                per_scene_avg[scene] = avg
            else:
                per_scene_avg[scene] = {"masked_clip_self": 0.0, "cross_similarity": 0.0,
                                        "net_isolation": 0.0, "global_scene_sim": 0.0, "count": 0}

        # 全局平均
        global_avg = {
            "masked_clip_self": round(np.mean([s["masked_clip_self"] for s in all_samples]), 4) if all_samples else 0.0,
            "cross_similarity": round(np.mean([s["cross_similarity"] for s in all_samples]), 4) if all_samples else 0.0,
            "net_isolation": round(np.mean([s["net_isolation"] for s in all_samples]), 4) if all_samples else 0.0,
            "global_scene_sim": round(np.mean([s["global_scene_sim"] for s in all_samples]), 4) if all_samples else 0.0,
            "total_samples": len(all_samples)
        }

        report[method] = {
            "per_scene_averages": per_scene_avg,
            "global_averages": global_avg
        }

        print(f"  全局平均: Self={global_avg['masked_clip_self']:.4f}, Cross={global_avg['cross_similarity']:.4f}, NetISO={global_avg['net_isolation']:.4f}, Scene={global_avg['global_scene_sim']:.4f}\n")

    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"增强评估完成，结果保存至 {SAVE_REPORT}")

if __name__ == "__main__":
    main()