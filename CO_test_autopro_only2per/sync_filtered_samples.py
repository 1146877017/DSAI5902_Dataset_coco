# 在co_dat_only2per\complete_samples_512\raw的基础上进行筛选，剔除裁剪后画面中不足2人的图片，得到co_dat_only2per\complete_samples_512_filter\raw，本代码用于同步后续的openpose、mask、depth_original_image、depth_pure_background

import os
import shutil
from tqdm import tqdm

# ===================== 路径配置 =====================
# 手工筛选结果所在目录
FILTER_BASE = r"co_dat_only2per/complete_samples_512_filter"
FILTER_RAW = os.path.join(FILTER_BASE, "raw")

# 原始的 512 数据源（包含所有模态）
SOURCE_BASE = r"co_dat_only2per/complete_samples_512"

# 需要同步的模态列表
MODALITIES = ["openpose", "depth_original_image", "depth_pure_background", "mask"]

# 1. 自动创建目标模态文件夹
for m in MODALITIES:
    os.makedirs(os.path.join(FILTER_BASE, m), exist_ok=True)

# 2. 获取手动筛选后保留的样本前缀
# 支持 .jpg 或 .png，取决于原始 raw 的格式
kept_prefixes = [os.path.splitext(f)[0] for f in os.listdir(FILTER_RAW) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"检测到筛选后保留的样本数量: {len(kept_prefixes)}")

# 3. 执行同步复制
for prefix in tqdm(kept_prefixes, desc="同步多模态特征"):
    for m in MODALITIES:
        # 在源目录中寻找对应的 png 文件
        src_file = os.path.join(SOURCE_BASE, m, f"{prefix}.png")
        dst_file = os.path.join(FILTER_BASE, m, f"{prefix}.png")
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"\n[Warning] 找不到对应的特征图: {src_file}")

print(f"\n全部模态同步完成！")