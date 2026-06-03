import os
import json
import numpy as np
import cv2
import torch
import clip
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# ===================== 配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
GEN_RESULTS = "synthetic_results"
SCENES = ["side_by_side", "handshake", "front_back"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 消融实验分组
ABLATION_GROUPS = {
    "baseline1": "纯文本",
    "baseline2": "仅OpenPose",
    "baseline3": "双ControlNet+无掩码",
    "ablation1": "双ControlNet+随机掩码",
    "ablation2": "仅OpenPose+实例掩码",
    "method": "双ControlNet+实例掩码"
}
SUFFIX_MAP = {
    "baseline1": "_baseline1",
    "baseline2": "_baseline2",
    "baseline3": "_baseline3",
    "ablation1": "_ablation1",
    "ablation2": "_ablation2",
    "method": "_method"
}

# 加载CLIP
model, preprocess = clip.load("ViT-B32", device=DEVICE)
SAVE_REPORT = "ablation_study_report.json"

# ===================== 评估函数 =====================
def compute_masked_clip(img_path, mask_path, prompt):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512,512))
    mask = cv2.resize(mask, (512,512))

    p1 = prompt.split("person1:")[1].split("person2:")[0].strip()
    p2 = prompt.split("person2:")[1].split(",")[0].strip()
    texts = clip.tokenize([p1,p2]).to(DEVICE)

    with torch.no_grad():
        text_emb = model.encode_text(texts)

    scores = []
    for pid in [1,2]:
        m = (mask==pid).astype(np.uint8)
        crop = cv2.bitwise_and(img,img,mask=m)
        crop = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        img_emb = model.encode_image(preprocess(crop).unsqueeze(0).to(DEVICE))
        sim = torch.cosine_similarity(img_emb, text_emb[pid-1:pid]).item()
        scores.append(sim)
    return np.mean(scores)

def compute_consistency(img_path1, img_path2):
    # 背景一致性SSIM
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return ssim(img1,img2)

# ===================== 消融实验 =====================
def run_ablation():
    print(" 开始运行消融实验\n")

    # 加载prompt
    with open(f"{SYNTHETIC_DATA}/synthetic_configs.json") as f:
        prompts = {c["scene"]:c["prompt"] for c in json.load(f)}

    report = {}

    # 遍历
    for group_name, group_desc in ABLATION_GROUPS.items():
        suffix = SUFFIX_MAP[group_name]
        print(f" 评估: {group_name} ({group_desc})")

        clip_scores = []
        consist_scores = []

        # 遍历场景计算指标
        for scene in SCENES:
            img_path = f"{GEN_RESULTS}/{scene}{suffix}.png"
            mask_path = f"{SYNTHETIC_DATA}/masks/{scene}.png"

            # 计算Masked CLIP
            clip_score = compute_masked_clip(img_path, mask_path, prompts[scene])
            clip_scores.append(clip_score)

            # 计算相邻场景叙事一致性
            if scene != SCENES[-1]:
                next_scene = SCENES[SCENES.index(scene)+1]
                img_next = f"{GEN_RESULTS}/{next_scene}{suffix}.png"
                consist = compute_consistency(img_path, img_next)
                consist_scores.append(consist)

        # 平均得分
        avg_clip = round(float(np.mean(clip_scores)),4)
        avg_consist = round(float(np.mean(consist_scores)),4) if consist_scores else 0

        report[group_name] = {
            "description": group_desc,
            "masked_clip": avg_clip,
            "narrative_consistency": avg_consist,
            "overall_score": round((avg_clip + avg_consist)/2,4)
        }

        print(f"   CLIP得分: {avg_clip:.4f} | 一致性: {avg_consist:.4f} | 总分: {report[group_name]['overall_score']:.4f}\n")

    # 保存
    with open(SAVE_REPORT,"w",encoding="utf-8") as f:
        json.dump(report,f,indent=4,ensure_ascii=False)

    print(" 消融实验完成！结果已保存！")

if __name__ == "__main__":
    run_ablation()