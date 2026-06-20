import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from controlnet_aux import OpenposeDetector
from transformers import CLIPProcessor, CLIPModel
from config import *

# 初始化评估模型（全局加载一次）
pose_detector = None
clip_model = None
clip_processor = None

def init_eval_models():
    global pose_detector, clip_model, clip_processor
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def evaluate_layout(condition_dir, gen_img_path):
    """评估布局精度：姿态关键点准确率 + 深度一致性"""
    input_pose = Image.open(os.path.join(condition_dir, "pose.png"))
    input_depth = cv2.imread(os.path.join(condition_dir, "depth.png"), cv2.IMREAD_GRAYSCALE)
    gen_img = Image.open(gen_img_path)

    # 1. 姿态关键点匹配率（基于二值图IOU近似）
    gen_pose = pose_detector(gen_img)
    input_pose_gray = np.array(input_pose.convert("L"))
    gen_pose_gray = np.array(gen_pose.convert("L"))

    inter = np.sum(np.logical_and(input_pose_gray > 128, gen_pose_gray > 128))
    union = np.sum(np.logical_or(input_pose_gray > 128, gen_pose_gray > 128))
    pose_acc = inter / union if union > 0 else 0.0

    # 2. 深度一致性：RMSE + SSIM
    # 此处使用输入深度图作为基准；实际可替换为 Depth-Anything 反推生成图深度
    gen_depth = np.array(gen_img.convert("L"))
    gen_depth_resized = cv2.resize(gen_depth, (input_depth.shape[1], input_depth.shape[0]))

    input_norm = input_depth / 255.0
    gen_norm = gen_depth_resized / 255.0

    depth_rmse = np.sqrt(np.mean((input_norm - gen_norm) ** 2))
    depth_ssim = ssim(input_norm, gen_norm, data_range=1.0)

    return {
        "pose_accuracy": pose_acc,
        "depth_rmse": depth_rmse,
        "depth_ssim": depth_ssim
    }

def evaluate_feature_isolation(condition_dir, gen_img_path, char_a_name, char_b_name):
    """评估特征隔离：Masked CLIP 相似度"""
    mask_a = cv2.imread(os.path.join(condition_dir, "mask_a.png"), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(os.path.join(condition_dir, "mask_b.png"), cv2.IMREAD_GRAYSCALE)
    gen_img = cv2.imread(gen_img_path)
    gen_img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)

    # 调整掩码尺寸并裁剪角色区域
    h, w = gen_img.shape[:2]
    mask_a_r = cv2.resize(mask_a, (w, h)) / 255.0
    mask_b_r = cv2.resize(mask_b, (w, h)) / 255.0

    char_a_img = gen_img_rgb * mask_a_r[..., np.newaxis]
    char_b_img = gen_img_rgb * mask_b_r[..., np.newaxis]

    # 角色对应文本描述
    char_a = CHARACTER_CONFIGS[char_a_name]
    char_b = CHARACTER_CONFIGS[char_b_name]
    text_a = f"{char_a['trigger']}, {char_a['desc']}"
    text_b = f"{char_b['trigger']}, {char_b['desc']}"

    # 计算CLIP嵌入
    inputs = clip_processor(
        text=[text_a, text_b],
        images=[char_a_img.astype(np.uint8), char_b_img.astype(np.uint8)],
        return_tensors="pt",
        padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = clip_model(**inputs)

    img_embeds = F.normalize(outputs.image_embeds, dim=1)
    text_embeds = F.normalize(outputs.text_embeds, dim=1)

    # 正确匹配相似度
    correct_a = (img_embeds[0] @ text_embeds[0]).item()
    correct_b = (img_embeds[1] @ text_embeds[1]).item()

    # 错误匹配相似度（特征泄露程度）
    wrong_a = (img_embeds[0] @ text_embeds[1]).item()
    wrong_b = (img_embeds[1] @ text_embeds[0]).item()

    # 隔离得分：正确相似度 - 错误相似度
    isolation_score = (correct_a + correct_b - wrong_a - wrong_b) / 2

    return {
        "correct_sim_avg": (correct_a + correct_b) / 2,
        "wrong_sim_avg": (wrong_a + wrong_b) / 2,
        "isolation_score": isolation_score
    }

def main():
    init_eval_models()
    methods = ["baseline1", "baseline2", "baseline3", "proposed"]
    all_results = []

    for method in methods:
        print(f"Evaluating {method} ...")
        for scene in SCENES:
            for bg in BACKGROUNDS:
                for (char_a, char_b) in CHARACTER_PAIRS:
                    condition_dir = os.path.join(CONDITION_DIR, scene, bg, f"{char_a}_{char_b}")
                    gen_path = os.path.join(OUTPUT_ROOT, method, scene, bg, f"{char_a}_{char_b}", "result.png")

                    if not os.path.exists(gen_path):
                        continue

                    layout_metrics = evaluate_layout(condition_dir, gen_path)
                    feat_metrics = evaluate_feature_isolation(condition_dir, gen_path, char_a, char_b)

                    row = {
                        "method": method,
                        "scene": scene,
                        "background": bg,
                        "char_pair": f"{char_a}_{char_b}",
                        **layout_metrics,
                        **feat_metrics
                    }
                    all_results.append(row)

    # 保存汇总结果
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_ROOT, "evaluation_summary.csv"), index=False)

    # 打印各方法平均指标
    print("\n===== Average Metrics by Method =====")
    print(df.groupby("method")[["pose_accuracy", "depth_ssim", "isolation_score"]].mean())
    print(f"\nFull report saved to: {OUTPUT_ROOT}/evaluation_summary.csv")

if __name__ == "__main__":
    main()