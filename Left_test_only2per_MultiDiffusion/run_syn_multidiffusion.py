# run_syn_multidiffusion_all90.py
import os
import json
import cv2
import subprocess
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ===================== 全局配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"        # 数据集根目录
OUTPUT_DIR = "synthetic_results"                 # 生成图像的输出目录
MD_SCRIPT = "MultiDiffusion-master/region_based.py"
TEMP_MASK_DIR = "temp_masks"

# 目标评测维度
TARGET_SCENES = ["side_by_side_stand", "side_by_side_raise_hand", "side_by_side_point", "handshake", "front_back"]
TARGET_CHARACTERS = ["Sera", "TogaHimiko", "MouriRan", "Byakuya"]

os.makedirs(TEMP_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 数据集读取与精确过滤 =====================
config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
print(f"[INIT] Loading synthetic configs from: {config_path}")
with open(config_path, "r", encoding="utf-8") as f:
    all_cfgs = json.load(f)

seen_conditions = set()
unique_cfgs = []

print("[INIT] Filtering dataset for complete 90-group matrix...")
for c in all_cfgs:
    scene = c["scene"]
    chars = sorted(c["characters"])  # 排序以规范化组合判定
    bg = c["background"]
    
    # 严格匹配目标场景与目标人物
    if scene in TARGET_SCENES and all(ch in TARGET_CHARACTERS for ch in chars):
        # 建立 场景-角色对-背景 的唯一性 Key
        condition_key = (scene, tuple(chars), bg)
        
        if condition_key not in seen_conditions:
            seen_conditions.add(condition_key)
            unique_cfgs.append(c)

total_tasks = len(unique_cfgs)
print(f"============================================================")
print(f"[INFO] Matrix matched: Found {total_tasks} unique experimental groups.")
print(f"[INFO] Target expectation: 5 scenes * 6 char-pairs * 3 bgs = 90 groups.")
print(f"============================================================")

# ===================== 主批处理生成逻辑 =====================
for idx, cfg in enumerate(unique_cfgs):
    sid = cfg["sample_id"]
    prompt = cfg["prompt"]
    scene = cfg["scene"]
    bg = cfg["background"]
    c1, c2 = cfg["characters"]
    
    out_path = os.path.join(OUTPUT_DIR, f"{sid}_multidiffusion.png")
    
    #  断点续传机制：如果文件存在则直接跳过
    if os.path.exists(out_path):
        print(f"[{idx+1}/{total_tasks}] Skip {sid} (Already exists)")
        continue
    
    print(f"\n[{idx+1}/{total_tasks}] Processing Sample: {sid}")
    print(f" -> Scene: {scene} | Background: {bg}")
    print(f" -> Characters: {c1} & {c2}")

    # 提取人物描述
    try:
        p1_desc = prompt.split("person1:")[1].split(". person2:")[0].strip()
        p2_desc = prompt.split("person2:")[1].split(". front view")[0].strip()
    except Exception:
        p1_desc, p2_desc = c1, c2
    
    # 背景描述转换
    bg_desc = bg.replace("_", " ") + ", " + scene.replace("_", " ")

    # 分离实例 Mask 为 MultiDiffusion 需要的二值化 Mask (0 或 255)
    src_mask = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")
    t1 = os.path.join(TEMP_MASK_DIR, f"{sid}_p1.png")
    t2 = os.path.join(TEMP_MASK_DIR, f"{sid}_p2.png")
    
    mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f" [ERROR] Mask file not found: {src_mask}, skipping.")
        continue
        
    cv2.imwrite(t1, np.where(mask == 128, 255, 0).astype(np.uint8))
    cv2.imwrite(t2, np.where(mask == 255, 255, 0).astype(np.uint8))

    # 构建 MultiDiffusion 运行时命令 
    cmd = [
        "python", MD_SCRIPT,
        "--mask_paths", t1, t2,
        "--bg_prompt", bg_desc,
        "--fg_prompts", p1_desc, p2_desc,
        "--sd_version", "1.5",
        "--H", "512", "--W", "512",
        "--seed", "42",
        "--steps", "25"
    ]
    
    try:
        # 运行子进程并实时抛出潜在错误
        subprocess.run(cmd, check=True)
        
        # 转移生成的图片至结果目录
        if os.path.exists("out.png"):
            os.rename("out.png", out_path)
            print(f" [SUCCESS] Generated and saved to: {out_path}")
        else:
            print(f" [WARNING] MultiDiffusion finished but 'out.png' was not found.")
            
    except subprocess.CalledProcessError as e:
        print(f" [ERR] MultiDiffusion script returned an error on {sid}: {e}")
    except Exception as e:
        print(f" [ERR] Failed to process sample {sid}: {e}")

print("\n============================================================")
print(" All 90 MultiDiffusion full-matrix experiments completed!")
print("============================================================")