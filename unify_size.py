import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = r"coco_multi_person/complete_samples"  
output_dir = r"coco_multi_person/complete_samples_512"  
target_size = (512, 512)  # 统一尺寸

# 输出
for subdir in ["raw", "openpose", "depth_original_image", "depth_pure_background", "mask"]:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def smart_crop_and_resize(img, target_size=512, interpolation=cv2.INTER_LINEAR):
    """
    智能裁剪：以长边为基准裁剪为正方形，再resize到目标尺寸，无黑边
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
    img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=interpolation)
    
    return img_resized

# 处理
sample_names = [f for f in os.listdir(os.path.join(input_dir, "raw")) if f.endswith(('.jpg', '.png'))]
for sample_name in tqdm(sample_names, desc="统一尺寸并智能裁剪"):
    sample_prefix = os.path.splitext(sample_name)[0]
    
    # 1. 处理原图 
    raw_path = os.path.join(input_dir, "raw", sample_name)
    raw_img = cv2.imread(raw_path)
    if raw_img is not None:
        raw_img_512 = smart_crop_and_resize(raw_img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "raw", sample_name), raw_img_512)
    
    # 2. 处理 OpenPose 图，强制保存为 .png
    openpose_path = os.path.join(input_dir, "openpose", sample_prefix + ".png")
    if os.path.exists(openpose_path):
        openpose_img = cv2.imread(openpose_path)
        openpose_img_512 = smart_crop_and_resize(openpose_img, target_size, interpolation=cv2.INTER_LINEAR)
        # 此处文件名改为 sample_prefix + ".png"
        cv2.imwrite(os.path.join(output_dir, "openpose", sample_prefix + ".png"), openpose_img_512)
    
    # 3. 处理 Depth_original_image 图，强制保存为 .png
    depth_original_image_path = os.path.join(input_dir, "depth_original_image", sample_prefix + ".png")
    if os.path.exists(depth_original_image_path):
        depth_original_image_img = cv2.imread(depth_original_image_path, cv2.IMREAD_GRAYSCALE)
        depth_original_image_img_512 = smart_crop_and_resize(depth_original_image_img, target_size, interpolation=cv2.INTER_LINEAR)
        # 此处文件名改为 sample_prefix + ".png"
        cv2.imwrite(os.path.join(output_dir, "depth_original_image", sample_prefix + ".png"), depth_original_image_img_512)
            
        
    # 4. 处理 Depth_pure_background 图，强制保存为 .png
    depth_pure_background_path = os.path.join(input_dir, "depth_pure_background", sample_prefix + ".png")
    if os.path.exists(depth_pure_background_path):
        depth_pure_background_img = cv2.imread(depth_pure_background_path, cv2.IMREAD_GRAYSCALE)
        depth_pure_background_img_512 = smart_crop_and_resize(depth_pure_background_img, target_size, interpolation=cv2.INTER_LINEAR)
        # 此处文件名改为 sample_prefix + ".png"
        cv2.imwrite(os.path.join(output_dir, "depth_pure_background", sample_prefix + ".png"), depth_pure_background_img_512)

    # 5. 处理 Mask 图，强制保存为 .png
    mask_path = os.path.join(input_dir, "mask", sample_prefix + ".png")
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Mask 的插值必须是最近邻插值 cv2.INTER_NEAREST，防止缩放产生浮点过渡值
        mask_img_512 = smart_crop_and_resize(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
        # 此处文件名改为 sample_prefix + ".png"
        cv2.imwrite(os.path.join(output_dir, "mask", sample_prefix + ".png"), mask_img_512)

print(f" 尺寸统一完成，调整掩码插值，保存到 {output_dir}，尺寸： 512×512")