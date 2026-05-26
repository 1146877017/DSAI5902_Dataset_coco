import json
import os
from tqdm import tqdm
import random
import re

# ===================== 路径配置 =====================
CAPTIONS_PATH = "../annotations/captions_val2017.json"
IMAGE_DIR = "../coco_multi_person/complete_samples_512/raw"
OUTPUT_PATH = "coco_person_prompts.json"

# 固定随机种子
random.seed(42)

# ===================== 特征池 =====================
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray"]
CLOTHING = ["shirt", "t-shirt", "jacket", "hoodie", "sweater", "suit", "coat", "blouse", "vest", "cardigan"]
GENDERS = ["man", "woman", "boy", "girl"]
ACTIONS = ["standing", "walking", "sitting", "running", "talking"]
ACCESSORIES = ["hat", "glasses", "backpack", "watch", "scarf"]

# ===================== 数据加载 =====================
print("正在加载 COCO 标注文件...")
with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
    coco_captions = json.load(f)
annotations = coco_captions["annotations"]

id2captions = {}
for ann in annotations:
    img_id = ann["image_id"]
    cap = ann["caption"].strip()
    id2captions.setdefault(img_id, []).append(cap)

if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"未找到原始图像目录，请检查路径: {IMAGE_DIR}")
    
# 【修复 4】显式进行排序，保证在任何操作系统和文件系统下的生成顺序和属性绑定完全绝对一致
valid_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))])

file2id = {}
for f in valid_files:
    name_part = os.path.splitext(f)[0]
    digits = re.findall(r'\d+', name_part)
    if digits:
        file2id[f] = int(digits[-1])
    else:
        print(f" 警告: 无法从文件名 '{f}' 中解析出有效的 image_id，该样本将被跳过。")

def clean_caption(raw_caption):
    cleaned = re.sub(r"[^a-zA-Z\s']", " ", raw_caption)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# ===================== 空间解耦 Prompt 生成 =====================
final_prompts = []
for fname in tqdm(valid_files, desc="生成空间规范化 Prompt"):
    img_id = file2id.get(fname)
    if img_id is None:
        continue
        
    cap_list = id2captions.get(img_id)
    if cap_list:
        raw_cap = random.choice(cap_list)
        clean_cap = clean_caption(raw_cap)
        base_prefix = f"{clean_cap}," if clean_cap else "A high-quality professional photo,"
    else:
        base_prefix = "A high-quality professional photo of two people,"
    
    color1, color2 = random.sample(COLORS, 2)
    cloth1, cloth2 = random.sample(CLOTHING, 2)
    gender1, gender2 = random.sample(GENDERS, 2)
    action1, action2 = random.sample(ACTIONS, 2)
    accessory1, accessory2 = random.sample(ACCESSORIES, 2)
    
    prompt = (
        f"{base_prefix} "
        f"person 1 on the left: a {gender1} wearing a {color1} {cloth1}, {action1} and wearing a {accessory1}, "
        f"person 2 on the right: a {gender2} wearing a {color2} {cloth2}, {action2} and wearing a {accessory2}"
    )
    
    final_prompts.append({
        "file_name": fname,
        "image_id": img_id,
        "prompt": prompt,
        "person1_gt": f"a {gender1} wearing a {color1} {cloth1}",
        "person2_gt": f"a {gender2} wearing a {color2} {cloth2}"
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_prompts, f, indent=2, ensure_ascii=False)

print(f" 成功！已生成 {len(final_prompts)} 个Prompt。结果保存在: {OUTPUT_PATH}")