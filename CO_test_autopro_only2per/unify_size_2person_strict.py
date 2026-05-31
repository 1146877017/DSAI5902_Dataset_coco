import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = r"coco_multi_person/complete_samples"  
output_dir = r"coco_multi_person/complete_samples_512"  
target_size = (512, 512)  # 统一尺寸，仅用于显示或将来扩展

# 输出
for subdir in ["raw", "openpose", "depth_original_image", "depth_pure_background", "mask"]:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def smart_crop_and_resize(img, size=512, interpolation=cv2.INTER_LINEAR):
    """
    智能裁剪：以长边为基准裁剪为正方形，再resize到 size x size，无黑边
    """
    h, w = img.shape[:2]
    
    # 裁剪为正方形（居中裁剪）
    if h > w:
        crop_size = w
        y_start = (h - w) // 2
        img_cropped = img[y_start:y_start+crop_size, :]
    else:
        crop_size = h
        x_start = (w - h) // 2
        img_cropped = img[:, x_start:x_start+crop_size]
    
    # 直接resize到目标尺寸
    img_resized = cv2.resize(img_cropped, (size, size), interpolation=interpolation)
    return img_resized

# 处理
sample_names = [f for f in os.listdir(os.path.join(input_dir, "raw")) if f.endswith(('.jpg', '.png'))]
for sample_name in tqdm(sample_names, desc="统一尺寸并智能裁剪"):
    sample_prefix = os.path.splitext(sample_name)[0]
    
    # 1. 处理原图 
    raw_path = os.path.join(input_dir, "raw", sample_name)
    raw_img = cv2.imread(raw_path)
    if raw_img is not None:
        raw_img_512 = smart_crop_and_resize(raw_img, size=512, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "raw", sample_name), raw_img_512)
    
    # 2. 处理 OpenPose 图
    openpose_path = os.path.join(input_dir, "openpose", sample_prefix + ".png")
    if os.path.exists(openpose_path):
        openpose_img = cv2.imread(openpose_path)
        openpose_img_512 = smart_crop_and_resize(openpose_img, size=512, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "openpose", sample_prefix + ".png"), openpose_img_512)
    
    # 3. 处理 Depth_original_image 图
    depth_original_image_path = os.path.join(input_dir, "depth_original_image", sample_prefix + ".png")
    if os.path.exists(depth_original_image_path):
        depth_original_image_img = cv2.imread(depth_original_image_path, cv2.IMREAD_GRAYSCALE)
        depth_original_image_img_512 = smart_crop_and_resize(depth_original_image_img, size=512, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "depth_original_image", sample_prefix + ".png"), depth_original_image_img_512)
    
    # 4. 处理 Depth_pure_background 图
    depth_pure_background_path = os.path.join(input_dir, "depth_pure_background", sample_prefix + ".png")
    if os.path.exists(depth_pure_background_path):
        depth_pure_background_img = cv2.imread(depth_pure_background_path, cv2.IMREAD_GRAYSCALE)
        depth_pure_background_img_512 = smart_crop_and_resize(depth_pure_background_img, size=512, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "depth_pure_background", sample_prefix + ".png"), depth_pure_background_img_512)

    # 5. 处理 Mask 图（最近邻插值）
    mask_path = os.path.join(input_dir, "mask", sample_prefix + ".png")
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img_512 = smart_crop_and_resize(mask_img, size=512, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "mask", sample_prefix + ".png"), mask_img_512)

print(f" 尺寸统一完成，保存到 {output_dir}，尺寸：512×512")