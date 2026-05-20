import json
import os
from tqdm import tqdm
import random

# ===================== 路径 =====================
CAPTIONS_PATH = "../annotations/captions_val2017.json"  # COCO caption文件
IMAGE_DIR = "../coco_multi_person/complete_samples_512/raw"  # 双人图路径
OUTPUT_PATH = "coco_person_prompts.json"  # 输出prompt配置

# 加载COCO captions
with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
    coco_captions = json.load(f)

annotations = coco_captions["annotations"]
images = coco_captions["images"]

# 建立 image_id -> caption 映射
id2captions = {}  # 存储所有caption
for ann in annotations:
    img_id = ann["image_id"]
    cap = ann["caption"]
    if img_id not in id2captions:
        id2captions[img_id] = []
    id2captions[img_id].append(cap)

# 建立 image_id -> filename 映射
id2file = {}
for img in images:
    img_id = img["id"]
    file_name = img["file_name"]
    id2file[img_id] = file_name

# 读取1167张图片
valid_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
file2id = {f: int(f.split(".")[0]) for f in valid_files}

# 生成标准化prompt（强制加入person1/person2，适配注意力掩码）
final_prompts = []
for fname in tqdm(valid_files, desc="生成prompt"):
    img_id = file2id[fname]
    cap_list = id2captions.get(img_id, ["a photo of two people"])
    cap = random.choice(cap_list)
    prompt = f"{cap}, two people, person1, person2"
    final_prompts.append({
        "file_name": fname,
        "image_id": img_id,
        "prompt": prompt
    })

# 保存配置
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_prompts, f, indent=2, ensure_ascii=False)

print(f" 完成 , 生成 {len(final_prompts)} 个prompt，保存到 {OUTPUT_PATH}")