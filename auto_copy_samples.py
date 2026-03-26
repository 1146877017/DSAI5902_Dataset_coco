import os
import shutil
from tqdm import tqdm

#  路径
RAW_DIR = r"coco_multi_person/raw"
OPENPOSE_DIR = r"coco_multi_person/openpose"
DEPTH_DIR = r"coco_multi_person/depth"
MASK_DIR = r"coco_multi_person/mask"

# 目标完整样本路径
TARGET_BASE = r"coco_multi_person/complete_samples"
TARGET_RAW = os.path.join(TARGET_BASE, "raw")
TARGET_OPENPOSE = os.path.join(TARGET_BASE, "openpose")
TARGET_DEPTH = os.path.join(TARGET_BASE, "depth")
TARGET_MASK = os.path.join(TARGET_BASE, "mask")

# 创建文件夹
for dir_path in [TARGET_RAW, TARGET_OPENPOSE, TARGET_DEPTH, TARGET_MASK]:
    os.makedirs(dir_path, exist_ok=True)

#  1：提取所有文件的前缀名
def get_file_prefixes(folder_path, suffixes=('.jpg', '.png')):
    """获取文件夹内所有图片的前缀"""
    prefixes = set()
    for f in os.listdir(folder_path):
        if f.lower().endswith(suffixes):
            prefix = os.path.splitext(f)[0]
            prefixes.add(prefix)
    return prefixes

# 提取四个文件夹的前缀
raw_prefix = get_file_prefixes(RAW_DIR)
op_prefix = get_file_prefixes(OPENPOSE_DIR)
depth_prefix = get_file_prefixes(DEPTH_DIR)
mask_prefix = get_file_prefixes(MASK_DIR)

# 取交集：同时存在 原图+OpenPose+Depth+Mask 的样本
complete_prefixes = raw_prefix & op_prefix & depth_prefix & mask_prefix

print(f"🔍 筛选完成：共找到 {len(complete_prefixes)} 个完整样本")

# 2：复制文件 
def copy_files(src_folder, dst_folder, prefixes, src_suffix):
    """复制文件，根据前缀匹配复制"""
    for prefix in tqdm(prefixes, desc=f"复制 {dst_folder.split('/')[-1]}"):
        src_file = os.path.join(src_folder, f"{prefix}{src_suffix}")
        dst_file = os.path.join(dst_folder, f"{prefix}{src_suffix}")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)

copy_files(RAW_DIR, TARGET_RAW, complete_prefixes, src_suffix=".jpg")
copy_files(OPENPOSE_DIR, TARGET_OPENPOSE, complete_prefixes, src_suffix=".jpg")
copy_files(DEPTH_DIR, TARGET_DEPTH, complete_prefixes, src_suffix=".jpg")
copy_files(MASK_DIR, TARGET_MASK, complete_prefixes, src_suffix=".png")

print(f"\n✅ 全部复制完成，完整样本保存到：{TARGET_BASE}")
print(f"📂 文件夹结构：")
print(f"   {TARGET_RAW} → 305张原图")
print(f"   {TARGET_OPENPOSE} → 305张姿态图")
print(f"   {TARGET_DEPTH} → 305张深度图")
print(f"   {TARGET_MASK} → 305张实例掩码")