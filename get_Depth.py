import os
import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2

#  配置路径
raw_img_dir = r"coco_multi_person/raw"       
depth_output_dir = r"coco_multi_person/depth"

os.makedirs(depth_output_dir, exist_ok=True)

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},  # 超大模型
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},      # medium
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}        # small
}

# 2. 初始化模型：vitb
encoder = 'vitb'
model = DepthAnythingV2(**model_configs[encoder])

# 3. 加载本地权重
weight_path = r"depth_anything_v2_vitb.pth"
checkpoint = torch.load(weight_path, map_location='cpu')
model.load_state_dict(checkpoint)

# 4. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

processed_count = 0
failed_count = 0

# 统计图片文件
img_files = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"开始生成深度图（保留原始尺寸），共 {len(img_files)} 个图片文件")

for img_name in img_files:
    img_path = os.path.join(raw_img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 跳过：无法读取图像 → {img_name}")
        failed_count += 1
        continue
    
    # 推理
    with torch.no_grad():
        depth = model.infer_image(img)  # 输入原始尺寸图像
    
    # 转换为8位灰度图
    depth_8bit = (depth * 255).astype(np.uint8)
    
    # 保存深度图
    save_path = os.path.join(depth_output_dir, img_name)
    cv2.imwrite(save_path, depth_8bit)
    
    processed_count += 1
    if processed_count % 20 == 0:
        print(f"进度：已处理 {processed_count} 个文件")

print("\n✅ 深度图（原始尺寸）生成完成！")
print(f"📊 统计：成功 {processed_count} 个，失败 {failed_count} 个")
print(f"💾 深度图保存路径：{depth_output_dir}")