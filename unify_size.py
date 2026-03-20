import os
import cv2
import numpy as np
from tqdm import tqdm

# 配置路径
input_dir = r"coco_multi_person/complete_samples"  
output_dir = r"coco_multi_person/complete_samples_512"  
target_size = (512, 512)  # 统一尺寸

# 输出
for subdir in ["raw", "openpose", "depth", "mask"]:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def resize_with_padding(img, target_size, interpolation=cv2.INTER_LINEAR):
    """
    保持比例，并补黑边到目标尺寸
    raw/openpose/depth：INTER_LINEAR
    mask：INTER_NEAREST
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # 缩放比例
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # 缩放图像
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # 目标尺寸的黑底图，缩放后的图居中放置
    if len(img.shape) == 3:  # RGB 
        img_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else:  # depth/mask
        img_padded = np.zeros((target_h, target_w), dtype=np.uint8)
    
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    if len(img.shape) == 3:
        img_padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w, :] = img_resized
    else:
        img_padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img_resized
    
    return img_padded

# 处理
sample_names = [f for f in os.listdir(os.path.join(input_dir, "raw")) if f.endswith(('.jpg', '.png'))]
for sample_name in tqdm(sample_names, desc="统一尺寸，调整掩码插值"):
    sample_prefix = os.path.splitext(sample_name)[0]
    
    # 1. 调整原始图（连续值 → 线性插值）
    raw_path = os.path.join(input_dir, "raw", sample_name)
    raw_img = cv2.imread(raw_path)
    raw_img_512 = resize_with_padding(raw_img, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, "raw", sample_name), raw_img_512)
    
    # 2. 处理 OpenPose 图（连续值 → 线性插值）
    openpose_path = os.path.join(input_dir, "openpose", sample_name)
    if os.path.exists(openpose_path):
        openpose_img = cv2.imread(openpose_path)
        openpose_img_512 = resize_with_padding(openpose_img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "openpose", sample_name), openpose_img_512)
    
    # 3. 处理 Depth 图（连续值 → 线性插值）
    depth_path = os.path.join(input_dir, "depth", sample_name)
    if os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_img_512 = resize_with_padding(depth_img, target_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, "depth", sample_name), depth_img_512)
    
    # 4. 处理 Mask 图（离散标签 → 最近邻插值）
    mask_path = os.path.join(input_dir, "mask", f"{sample_prefix}.png")
    if os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img_512 = resize_with_padding(mask_img, target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, "mask", f"{sample_prefix}.png"), mask_img_512)

print(f"✅ 尺寸统一完成，调整掩码插值，保存到 {output_dir}，尺寸： 512×512")