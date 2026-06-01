import json
import os
from tqdm import tqdm
import random
import re
import cv2
import numpy as np

# ===================== 路径配置 =====================
CAPTIONS_PATH = "../annotations/captions_val2017.json"
IMAGE_DIR = "co_dat_only2per/complete_samples_512_filter/raw"         
MASK_DIR = "co_dat_only2per/complete_samples_512_filter/mask"       # 引入 mask 文件夹用于判断位置
OUTPUT_PATH = "coco_person_prompts.json"

# 固定随机种子确保实验可重复性
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
    cleaned = raw_caption.lower()
    cleaned = re.sub(r'\b(two people|two men|two women|couple|players)\b', 'people', cleaned)
    cleaned = re.sub(r"[^a-zA-Z\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# ===================== 空间对齐 Prompt 生成 =====================
final_prompts = []
for fname in tqdm(valid_files, desc="生成空间规范化 Prompt"):
    name_part = os.path.splitext(fname)[0]
    img_id = file2id.get(fname)
    if img_id is None:
        continue
        
    mask_path = os.path.join(MASK_DIR, name_part + ".png")
    
    # 严格检查 Mask 是否存在
    if not os.path.exists(mask_path):
        print(f" 警告: 未找到 Mask 文件 '{mask_path}'，该样本将被跳过。")
        continue

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    unique_vals = [v for v in np.unique(mask_img) if v > 0]
    
    # 严格筛选：必须刚好包含2个实例的图片才处理（符合 co_dat_only2per 的定义）
    if len(unique_vals) != 2:
        print(f" 警告: 样本 '{fname}' 的 Mask 中检测到 {len(unique_vals)} 个目标（期望为2），已被跳过。")
        continue

    # 计算两个实例在 X 轴上的几何中心
    x_center_0 = np.mean(np.where(mask_img == unique_vals[0])[1])
    x_center_1 = np.mean(np.where(mask_img == unique_vals[1])[1])
    
    # 显式映射：哪一个 Mask ID 是左边，哪一个是右边
    if x_center_0 < x_center_1:
        left_mask_id = int(unique_vals[0])
        right_mask_id = int(unique_vals[1])
    else:
        left_mask_id = int(unique_vals[1])
        right_mask_id = int(unique_vals[0])

    # 获取基础背景描述
    cap_list = id2captions.get(img_id)
    if cap_list:
        raw_cap = random.choice(cap_list)
        clean_cap = clean_caption(raw_cap)
        base_prefix = f"A realistic photo of {clean_cap}," if clean_cap else "A high-quality professional photo,"
    else:
        base_prefix = "A high-quality professional photo of people,"
    
    # 随机采样完全不同的属性
    g_left, g_right = random.sample(GENDERS, 2)
    c_left, c_right = random.sample(COLORS, 2)
    cl_left, cl_right = random.sample(CLOTHING, 2)
    act_left, act_right = random.sample(ACTIONS, 2)
    acc_left, acc_right = random.sample(ACCESSORIES, 2)

    # 拼接空间位置感知的 Prompt
    prompt = (
        f"{base_prefix} "
        f"the person on the left is a {g_left} wearing a {c_left} {cl_left}, {act_left} and wearing a {acc_left}, "
        f"the person on the right is a {g_right} wearing a {c_right} {cl_right}, {act_right} and wearing a {acc_right}"
    )
    
    # 保存结果，包含精确的 Mask ID 映射，方便下游干预实验直接读取
    final_prompts.append({
        "file_name": fname,
        "image_id": img_id,
        "prompt": prompt,
        "left_mask_id": left_mask_id,      # 左边人在 mask 图中的像素值
        "right_mask_id": right_mask_id,    # 右边人在 mask 图中的像素值
        "person_left_gt": f"a {g_left} wearing a {c_left} {cl_left}",
        "person_right_gt": f"a {g_right} wearing a {c_right} {cl_right}"
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_prompts, f, indent=2, ensure_ascii=False)

print(f" 成功！已生成 {len(final_prompts)} 个具备【空间位置及掩码感知】的 Prompt。结果保存在: {OUTPUT_PATH}")