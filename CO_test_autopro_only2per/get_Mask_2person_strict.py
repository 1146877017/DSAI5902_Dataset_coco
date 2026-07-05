import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# ===================== 路径配置 =====================
anno_path = r"../annotations/instances_val2017.json"  # COCO官方标注文件
raw_img_dir = r"co_dat_only2per/raw"
mask_output_dir = r"co_dat_only2per/mask"
os.makedirs(mask_output_dir, exist_ok=True)

# 加载COCO标注
coco = COCO(anno_path)
cat_id = coco.getCatIds(catNms=['person'])[0]  # 只取person类别

# 获取所有2人物图片的ID
with open(r"co_dat_only2per/multi_person_ids.json", "r", encoding="utf-8") as f:
    multi_person_ids = json.load(f)["image_ids"]

print(f"Start generating masks based on the official annotations from COCO, in total {len(multi_person_ids)} ")

for img_id in tqdm(multi_person_ids, desc="Generate the official COCO mask"):
    # 获取图片信息
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info["file_name"]
    H, W = img_info["height"], img_info["width"]
    
    # 获取该图片的所有person标注
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    
    # 生成实例掩码
    instance_mask = np.zeros((H, W), dtype=np.uint8)
    for idx, ann in enumerate(anns, start=1):
        # 从COCO标注生成二值掩码
        mask = coco.annToMask(ann)
        # 赋值为人物序号
        instance_mask[mask == 1] = idx
    
    # 保存为PNG
    if len(anns) > 0:
        save_path = os.path.join(mask_output_dir, os.path.splitext(img_name)[0] + ".png")
        cv2.imwrite(save_path, instance_mask)
    else:
        print(f" Warning: Image {img_id} does not have person annotations. Please check data consistency.")

print("\n COCO official instance masks generated!")