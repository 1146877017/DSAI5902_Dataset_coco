import os
import json
import cv2
import torch
import clip
import numpy as np
from PIL import Image

# ===================== 配置 =====================
# 合成测试集路径
SYNTHETIC_DATASET = "synthetic_test_dataset"
# 合成实验生成结果
GEN_RESULTS = "synthetic_results"
# 场景与方法
SCENES = ["side_by_side", "handshake", "front_back"]
METHODS = ["baseline1", "baseline2", "baseline3", "method"]
SUFFIXES = ["_baseline1", "_baseline2", "_baseline3", "_method"]
SAVE_REPORT = "proposal_masked_clip_report.json"

# 图像尺寸
IMAGE_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 加载 CLIP 模型 =====================
print(" 加载 CLIP 模型...")
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== Masked CLIP 计算 =====================
def get_person_descriptions_from_prompt(prompt):
    """
    从 prompt 拆分 person1 / person2 
    """
    prompt = prompt.lower()
    p1 = prompt.split("person1:")[1].split("person2:")[0].strip()
    p2 = prompt.split("person2:")[1].split(",")[0].strip()
    return [p1, p2]

def compute_proposal_masked_clip(img_path, mask_path, prompt):
    """
    Proposal 定义：
    1. 用掩码裁剪每个角色
    2. 图像Embedding vs 对应角色文本Embedding
    3. 余弦相似度衡量特征是否隔离成功
    """
    # 1. 读取生成图 + 掩码
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # 2. 拆分角色描述
    text_descriptions = get_person_descriptions_from_prompt(prompt)

    # 3. 文本编码
    text_tokens = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)

    similarities = []
    # 4. 各角色：掩码裁剪 → 图像编码 → 相似度
    for person_id in [1, 2]:
        # 提取单个角色掩码
        person_mask = (mask == person_id).astype(np.uint8)
        # 掩码裁剪
        masked_img = cv2.bitwise_and(img, img, mask=person_mask)
        # 转PIL & CLIP预处理
        masked_pil = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(masked_pil).unsqueeze(0).to(device)
        # 图像编码
        with torch.no_grad():
            img_embed = model.encode_image(img_tensor)
        # 对应角色文本相似度
        sim = torch.cosine_similarity(img_embed, text_embeds[person_id-1:person_id]).item()
        similarities.append(sim)

    # 两个角色的平均相似度
    return round(float(np.mean(similarities)), 4)

# ===================== 评估流程 =====================
def main():
    print("\n 开始 Masked CLIP 评估】")
    print("验证多人物特征是否隔离\n")

    # 加载所有场景的prompt
    with open(os.path.join(SYNTHETIC_DATASET, "synthetic_configs.json"), "r", encoding="utf-8") as f:
        scene_configs = { cfg["scene"]: cfg for cfg in json.load(f) }

    report = {}

    # 遍历所有方法
    for method, suffix in zip(METHODS, SUFFIXES):
        print(f" 评估方法：{method.upper()}")
        method_scores = []

        # 遍历所有场景
        for scene in SCENES:
            gen_img_path = os.path.join(GEN_RESULTS, f"{scene}{suffix}.png")
            mask_path = os.path.join(SYNTHETIC_DATASET, "masks", f"{scene}.png")
            prompt = scene_configs[scene]["prompt"]

            # 计算指标
            score = compute_proposal_masked_clip(gen_img_path, mask_path, prompt)
            method_scores.append(score)
            print(f"   {scene}: {score:.4f}")

        # 平均得分
        avg_score = round(float(np.mean(method_scores)), 4)
        report[method] = {
            "per_scene_scores": method_scores,
            "average_masked_clip_score": avg_score
        }
        print(f" {method} 最终得分：{avg_score:.4f}\n")

    # 保存
    with open(SAVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(" Masked CLIP 评估完成！")
    print(f"保存：{SAVE_REPORT}")
    print("\n 结论： 得分最高 → 特征隔离最成功！")

if __name__ == "__main__":
    main()