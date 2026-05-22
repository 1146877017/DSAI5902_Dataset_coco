import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

# ===================== 路径配置 =====================
raw_img_dir = r"coco_multi_person/raw"
depth_output_dir = r"coco_multi_person/depth"
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
checkpoint = torch.load(weight_path, map_location='cpu')
model.load_state_dict(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===================== 批量处理 =====================
img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"开始生成深度图（保留原始尺寸），共 {len(img_files)} 个文件")

processed_count = 0
failed_count = 0

for img_name in tqdm(img_files, desc="生成纯背景深度图"):
    img_path = os.path.join(raw_img_dir, img_name)
    mask_path = os.path.join(r"coco_multi_person/mask", os.path.splitext(img_name)[0] + ".png")
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        print(f"Err 无法读取图像或掩码 → {img_name}")
        failed_count += 1
        continue
    
    # 抹除人物区域，用背景修复填充
    inpaint_mask = (mask > 0).astype(np.uint8) * 255
    img_inpaint = cv2.inpaint(img, inpaint_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    
    # 对纯背景图生成深度
    with torch.no_grad():
        depth = model.infer_image(img_inpaint)
    
    # 归一化+保存
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max > depth_min:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)
    depth_8bit = (depth_norm * 255).astype(np.uint8)
    
    save_path = os.path.join(depth_output_dir, os.path.splitext(img_name)[0] + ".png")
    cv2.imwrite(save_path, depth_8bit)
    processed_count += 1

print(f"\n 深度图生成完成！成功：{processed_count}，失败：{failed_count}")
print(f"保存路径：{depth_output_dir}")