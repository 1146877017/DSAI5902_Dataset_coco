import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO

# 配置路径
raw_img_dir = r"coco_multi_person/raw"
mask_output_dir = r"coco_multi_person/mask"
os.makedirs(mask_output_dir, exist_ok=True)

#  1. 加载轻量模型
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 自动掩码生成参数
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    points_per_batch=64,
)

#  2. 加载 YOLO 检测人体框
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)

# 3. 批量处理图像
processed_count = 0
failed_count = 0
img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"开始生成语义 Mask（保留原始尺寸），共 {len(img_files)} 个图片文件")

for img_name in img_files:
    img_path = os.path.join(raw_img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 跳过：无法读取图像 → {img_name}")
        failed_count += 1
        continue
    H, W = img.shape[:2]

    # step1：YOLO 检测所有人体框
    yolo_results = yolo_model(img, classes=[0], verbose=False)
    person_boxes = []
    if yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes.xyxy.cpu().numpy():
            person_boxes.append(box)  # [x1, y1, x2, y2]

    if not person_boxes:
        print(f"⚠️ 未检测到人物 → {img_name}")
        failed_count += 1
        continue

    # step2：SAM 自动生成全图掩码
    masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # step3：过滤并绑定掩码到对应人体框
    person_mask_groups = [[] for _ in range(len(person_boxes))]
    for mask_dict in masks:
        mask = mask_dict["segmentation"]  # HxW bool
        mask_area = mask.sum()
        if mask_area < 1000:  # 过滤小干扰掩码
            continue

        # 计算与所有框的 IoU，匹配 IoU 最高的框
        max_iou = 0.3  
        best_box_idx = -1
        for box_idx, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box.astype(int)
            # 生成与图像同尺寸的 box_mask
            box_mask = np.zeros((H, W), dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            
            intersection = np.logical_and(mask, box_mask).sum()
            union = np.logical_or(mask, box_mask).sum()
            iou = intersection / (union + 1e-6)
            if iou > max_iou:
                max_iou = iou
                best_box_idx = box_idx
        if best_box_idx != -1:
            person_mask_groups[best_box_idx].append(mask)

    # step4：合并同框掩码,处理重叠
    # 初始化
    pixel_votes = np.zeros((H, W, 2), dtype=np.float32)  # [H, W, (box_idx, weight)]
    for box_idx, masks_in_box in enumerate(person_mask_groups, start=1):
        if not masks_in_box:
            continue
        # 合并同一个框内的所有掩码
        combined_mask = np.zeros((H, W), dtype=bool)
        for m in masks_in_box:
            combined_mask = np.logical_or(combined_mask, m)
        if combined_mask.sum() == 0:
            continue

        x1, y1, x2, y2 = person_boxes[box_idx-1].astype(int)
        box_mask = np.zeros((H, W), dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        intersection = np.logical_and(combined_mask, box_mask).sum()
        union = np.logical_or(combined_mask, box_mask).sum()
        weight = intersection / (union + 1e-6)  # IoU 作为权重

        # 矢量操作更新投票
        mask_y, mask_x = np.where(combined_mask)
        current_weights = pixel_votes[mask_y, mask_x, 1]
        update_mask = weight > current_weights
        # 只更新权重更高的像素
        pixel_votes[mask_y[update_mask], mask_x[update_mask], 0] = box_idx
        pixel_votes[mask_y[update_mask], mask_x[update_mask], 1] = weight

    # 生成最终 Mask
    instance_mask = pixel_votes[..., 0].astype(np.uint8)

    if instance_mask.sum() == 0:
        print(f"⚠️ 未分割到人物 → {img_name}")
        failed_count += 1
        continue

    # step5：保存为 PNG
    save_name = os.path.splitext(img_name)[0] + ".png"
    save_path = os.path.join(mask_output_dir, save_name)
    cv2.imwrite(save_path, instance_mask)

    processed_count += 1
    if processed_count % 20 == 0:
        print(f"进度：已处理 {processed_count} 个文件")

print("\n✅  Mask 生成完成！")
print(f"📊 统计：成功 {processed_count} 个，失败 {failed_count} 个")
print(f"💾 Mask 保存路径：{mask_output_dir}")