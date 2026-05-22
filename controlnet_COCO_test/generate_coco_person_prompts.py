import json
import os
from tqdm import tqdm
import random
import re  # 正则清理字符串

# ===================== 路径 =====================
CAPTIONS_PATH = "../annotations/captions_val2017.json"
IMAGE_DIR = "../coco_multi_person/complete_samples_512/raw"
OUTPUT_PATH = "coco_person_prompts.json"

# 固定随机种子保证可复现
random.seed(42)

# ===================== 特征池 =====================
# 1. 扩展服装列表，新增性别、动作、配饰维度
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray"]
CLOTHING = ["shirt", "t-shirt", "jacket", "hoodie", "sweater", "suit", "coat", "blouse", "vest", "cardigan"]
GENDERS = ["man", "woman", "boy", "girl"]  # 性别维度
ACTIONS = ["standing", "walking", "sitting", "running", "talking"]  # 动作维度
ACCESSORIES = ["hat", "glasses", "backpack", "watch", "scarf"]  # 配饰维度

# 加载COCO captions
with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
    coco_captions = json.load(f)
annotations = coco_captions["annotations"]
images = coco_captions["images"]

# 建立 image_id -> 所有caption 映射
id2captions = {}
for ann in annotations:
    img_id = ann["image_id"]
    cap = ann["caption"].strip()
    if img_id not in id2captions:
        id2captions[img_id] = []
    id2captions[img_id].append(cap)

# 建立 image_id -> filename 映射
id2file = {}
for img in images:
    img_id = img["id"]
    id2file[img_id] = img["file_name"]

# 读取双人图
valid_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
file2id = {f: int(os.path.splitext(f)[0]) for f in valid_files}

# ===================== 字符串清理函数 =====================
def clean_caption(raw_caption):
    """清理caption中的多余标点，仅保留字母、空格、单引号"""
    # 1. 移除所有非字母/空格/单引号的字符
    cleaned = re.sub(r"[^a-zA-Z\s']", "", raw_caption)
    # 2. 移除末尾的句点 + 多余空格
    cleaned = cleaned.strip().rstrip(".")
    # 3. 合并多个连续空格为单个
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

# ===================== 生成prompt =====================
final_prompts = []
for fname in tqdm(valid_files, desc="生成prompt"):
    img_id = file2id[fname]
    # 随机取一条COCO官方caption
    cap_list = id2captions.get(img_id, ["a photo of two people"])
    raw_cap = random.choice(cap_list)
    
    # 调用清理函数：彻底清理标点
    clean_cap = clean_caption(raw_cap)
    
    # ===================== 特征全差异化 =====================
    # 1. 颜色：保证person1和person2不同
    color1, color2 = random.sample(COLORS, 2)
    # 2. 服装：保证person1和person2不同
    cloth1, cloth2 = random.sample(CLOTHING, 2)
    # 3. 维度差异化
    gender1, gender2 = random.sample(GENDERS, 2)
    action1, action2 = random.sample(ACTIONS, 2)
    accessory1, accessory2 = random.sample(ACCESSORIES, 2)
    
    # 构造prompt
    prompt = (
        f"{clean_cap}, "
        f"person 1: {gender1} wearing a {color1} {cloth1}, {action1} with a {accessory1}, "
        f"person 2: {gender2} wearing a {color2} {cloth2}, {action2} with a {accessory2}"
    )
    
    final_prompts.append({
        "file_name": fname,
        "image_id": img_id,
        "prompt": prompt
    })

# 保存
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_prompts, f, indent=2, ensure_ascii=False)

print(f"完成！生成 {len(final_prompts)} 个prompt，特征差异化")