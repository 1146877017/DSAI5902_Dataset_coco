import json
import os
import shutil
from collections import defaultdict

anno_path = r"annotations\instances_val2017.json"
raw_img_dir = r"val2017"
output_dir = r"coco_multi_person\raw"
max_samples = 500
# ID列表保存路径
id_list_save_path = r"coco_multi_person\multi_person_ids.json"

# 1. 加载标注+统计person数量
with open(anno_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)
person_count = defaultdict(int)
for ann in coco_data["annotations"]:
    if ann["category_id"] == 1:
        person_count[ann["image_id"]] += 1

# 2. 筛选多角色ID
multi_person_ids = [img_id for img_id, cnt in person_count.items() if cnt >= 2][:max_samples]
print(f"筛选出包含 ≥2 个 person 的样本数：{len(multi_person_ids)}")

# 3. 保存ID列表
os.makedirs(os.path.dirname(id_list_save_path), exist_ok=True)
with open(id_list_save_path, "w", encoding="utf-8") as f:
    json.dump({
        "筛选规则": "COCO 2017 Val集，person数量≥2，最多1000个样本",
        "image_ids": multi_person_ids,
        "样本数量": len(multi_person_ids)
    }, f, indent=4)  # indent=4 让JSON更易读
print(f"✅ 筛选出的图像ID列表已保存至：{id_list_save_path}")

# 4. 复制图像
os.makedirs(output_dir, exist_ok=True)
for img_id in multi_person_ids:
    img_filename = f"{img_id:012d}.jpg"
    src_path = os.path.join(raw_img_dir, img_filename)
    dst_path = os.path.join(output_dir, img_filename)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
print(f"筛选完成！所有多角色图像已保存到：{output_dir}")