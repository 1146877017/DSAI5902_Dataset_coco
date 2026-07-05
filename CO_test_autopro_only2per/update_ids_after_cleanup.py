# 算法粗筛后进行人工质检后，再次生成co_dat_only2per\multi_person_ids.json

import os
import json

# ===================== 路径配置  =====================
raw_img_dir = r"co_dat_only2per/raw"
id_list_save_path = r"co_dat_only2per/multi_person_ids.json"

# 1. 检查目标目录是否存在
if not os.path.exists(raw_img_dir):
    print(f" Error: Directory {os.path.abspath(raw_img_dir)} not found, please ensure you are running this script in the correct folder.")
    exit()

# 2. 读取手动筛选后，真正保留下来的图片文件名
remaining_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 3. 从文件名反向解析出标准的 COCO image_id (例如 "000000123456.jpg" -> 123456)
updated_image_ids = []
for filename in remaining_files:
    try:
        # 移除后缀，并将前导零字符串转换为纯数字整数 ID
        img_id = int(os.path.splitext(filename)[0])
        updated_image_ids.append(img_id)
    except ValueError:
        print(f" Warning: File [{filename}] cannot be parsed as a standard COCO ID, skipping.")

# 对 ID 进行排序，保持 JSON 数据的有序
updated_image_ids.sort()
actual_count = len(updated_image_ids)

print(f" Scan completed: After manual cleaning, there are {actual_count} valid images remaining in the current folder.")

# 4. 重新写入覆盖原有的 multi_person_ids.json
with open(id_list_save_path, "w", encoding="utf-8") as f:
    json.dump({
        "Filtering rules": "COCO 2017 Val set, with 2 persons, and the blurry/indistinct samples have been manually inspected and removed",
        "image_ids": updated_image_ids,
        "sample count": actual_count
    }, f, indent=4, ensure_ascii=False)

print(f" Success! The updated ID list has been synchronized to: {id_list_save_path}")
