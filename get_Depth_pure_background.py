import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

# ===================== 模式与路径配置 =====================
# ['pure_background', 'original_image'] 
# pure_background: 抹除人物（用于解耦实验） | original_image: 直出深度（用于保持ControlNet一致性）
MODE = 'pure_background'  

raw_img_dir = r"coco_multi_person/raw"
mask_dir = r"coco_multi_person/mask"

# 根据模式动态创建输出文件夹
depth_output_dir = f"coco_multi_person/depth_{MODE}"
os.makedirs(depth_output_dir, exist_ok=True)

# ===================== 模型配置 =====================
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}
encoder = 'vitb'  
model = DepthAnythingV2(**model_configs[encoder])

weight_path = r"depth_anything_v2_vitb.pth"
if not os.path.exists(weight_path):
    print(f" 错误：未找到权重文件：{weight_path}")
    exit(1)

checkpoint = torch.load(weight_path, map_location='cpu')
model.load_state_dict(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===================== 批量处理 =====================
img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f" 当前运行模式：【{MODE}】| 目标文件数：{len(img_files)}")

processed_count = 0
failed_count = 0

for img_name in tqdm(img_files, desc=f"生成深度图({MODE})"):
    img_path = os.path.join(raw_img_dir, img_name)
    mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            failed_count += 1
            continue
            
        # ===================== 根据分支选择前处理 =====================
        if MODE == 'pure_background':
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f" 模式为纯背景，但未找到对应的 Mask → {img_name}")
                failed_count += 1
                continue
                
            # 1. 基础二值化
            inpaint_mask = (mask > 0).astype(np.uint8) * 255
            
            # 2. 温和的形态学膨胀：用 5x5 的核，迭代 1 次（约向外扩张 5 像素），防止过度吞噬背景
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            inpaint_mask_dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)
            
            # 3. 适度的结构化修复：降低半径到 6 像素，紧贴外围边缘修复，减少大面积模糊
            img_input = cv2.inpaint(img, inpaint_mask_dilated, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
        else:
            # original_image 模式：直接将原图送入模型，不进行任何抹除
            img_input = img.copy()
        
        # ===================== 生成深度图 =====================
        with torch.no_grad():
            depth = model.infer_image(img_input)
        
        # 归一化
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        
        # 保存结果
        save_path = os.path.join(depth_output_dir, os.path.splitext(img_name)[0] + ".png")
        cv2.imwrite(save_path, depth_8bit)
        processed_count += 1
        
    except Exception as e:
        print(f" 处理 {img_name} 失败，原因：{str(e)}")
        failed_count += 1

print(f"\n 模式【{MODE}】处理完成！成功：{processed_count} | 失败：{failed_count}")
print(f" 保存路径：{depth_output_dir}")