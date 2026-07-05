import os
import shutil
from tqdm import tqdm

# ===================== 路径配置 =====================
RAW_DIR = r"co_dat_only2per/raw"
OPENPOSE_DIR = r"co_dat_only2per/openpose"
DEPTH_original_image_DIR = r"co_dat_only2per/depth_original_image"
DEPTH_pure_background_DIR = r"co_dat_only2per/depth_pure_background"
MASK_DIR = r"co_dat_only2per/mask"

# 目标完整样本路径
TARGET_BASE = r"co_dat_only2per/complete_samples"
TARGET_RAW = os.path.join(TARGET_BASE, "raw")
TARGET_OPENPOSE = os.path.join(TARGET_BASE, "openpose")
TARGET_DEPTH_original_image = os.path.join(TARGET_BASE, "depth_original_image")
TARGET_DEPTH_pure_background = os.path.join(TARGET_BASE, "depth_pure_background")
TARGET_MASK = os.path.join(TARGET_BASE, "mask")

# 创建文件夹
for dir_path in [TARGET_RAW, TARGET_OPENPOSE, TARGET_DEPTH_original_image, TARGET_DEPTH_pure_background, TARGET_MASK]:
    os.makedirs(dir_path, exist_ok=True)

# 1：提取所有文件的前缀名
def get_file_prefixes(folder_path, suffixes=('.jpg', '.jpeg', '.png')):
    """获取文件夹内所有图片的前缀"""
    prefixes = set()
    if not os.path.exists(folder_path):
        return prefixes
    for f in os.listdir(folder_path):
        if f.lower().endswith(suffixes):
            prefix = os.path.splitext(f)[0]
            prefixes.add(prefix)
    return prefixes

# 提取五个文件夹的前缀
raw_prefix = get_file_prefixes(RAW_DIR)
op_prefix = get_file_prefixes(OPENPOSE_DIR)
depth_original_image_prefix = get_file_prefixes(DEPTH_original_image_DIR)
depth_pure_background_prefix = get_file_prefixes(DEPTH_pure_background_DIR)
mask_prefix = get_file_prefixes(MASK_DIR)

# 取交集：同时存在 原图+OpenPose+Depth+Mask 的完美对齐样本
complete_prefixes = raw_prefix & op_prefix & depth_original_image_prefix & depth_pure_background_prefix & mask_prefix

print(f"  Screening alignment completed: The total number of perfectly aligned samples common to all 5 categories is:{len(complete_prefixes)} ")

# 2：复制文件 
def copy_files(src_folder, dst_folder, prefixes, src_suffix):
    """复制文件，根据前缀匹配复制，并保持元数据"""
    # 
    folder_name = os.path.basename(dst_folder) 
    for prefix in tqdm(prefixes, desc=f"正在同步  {folder_name}"):
        src_file = os.path.join(src_folder, f"{prefix}{src_suffix}")
        dst_file = os.path.join(dst_folder, f"{prefix}{src_suffix}")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)

# 批量安全同步
copy_files(RAW_DIR, TARGET_RAW, complete_prefixes, src_suffix=".jpg")
copy_files(OPENPOSE_DIR, TARGET_OPENPOSE, complete_prefixes, src_suffix=".png")
copy_files(DEPTH_original_image_DIR, TARGET_DEPTH_original_image, complete_prefixes, src_suffix=".png")
copy_files(DEPTH_pure_background_DIR, TARGET_DEPTH_pure_background, complete_prefixes, src_suffix=".png")
copy_files(MASK_DIR, TARGET_MASK, complete_prefixes, src_suffix=".png")

print(f"\n  All multimodal features have been aligned and synchronized successfully. Saved as: {TARGET_BASE}")