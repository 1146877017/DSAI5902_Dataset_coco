import json
import os
import shutil
from collections import defaultdict

# ===================== 路径 =====================
anno_path = r"annotations\instances_val2017.json"       # COCO标注文件
raw_img_dir = r"val2017"                                 # 原始图片
output_dir = r"coco_multi_person\raw"                    # 输出多人物图片
max_samples = 5000                                      # 最大筛选5000个样本
id_list_save_path = r"coco_multi_person\multi_person_ids.json"  # 保存ID列表

# 1. 加载标注 + 统计每个图片的人物数量
with open(anno_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

person_count = defaultdict(int)
for ann in coco_data["annotations"]:
    # COCO category_id=1 代表 person
    if ann["category_id"] == 1:
        person_count[ann["image_id"]] += 1

# 2. 筛选包含 ≥2 个人的图片ID
multi_person_ids = [img_id for img_id, cnt in person_count.items() if cnt >= 2][:max_samples]
print(f" 筛选出包含 ≥2 个 person 的样本数：{len(multi_person_ids)}")

# 3. 保存筛选后的ID列表
os.makedirs(os.path.dirname(id_list_save_path), exist_ok=True)
with open(id_list_save_path, "w", encoding="utf-8") as f:
    json.dump({
        "筛选规则": "COCO 2017 Val集，person数量≥2，最多5000个样本",
        "image_ids": multi_person_ids,
        "样本数量": len(multi_person_ids)
    }, f, indent=4)  
print(f" 图像ID列表已保存至：{id_list_save_path}")

# 4. 复制图片到目标文件夹
os.makedirs(output_dir, exist_ok=True)
copy_count = 0
for img_id in multi_person_ids:
    img_filename = f"{img_id:012d}.jpg"
    src_path = os.path.join(raw_img_dir, img_filename)
    dst_path = os.path.join(output_dir, img_filename)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copy_count += 1

# 打印结果
print(f" 图片复制完成！")
print(f" 实际复制有效样本数：{copy_count} / 筛选总数：{len(multi_person_ids)}")
print(f" 所有多人物图像保存至：{output_dir}")