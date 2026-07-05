import os

# 路径配置
raw_dir = r"co_dat_only2per/raw"
openpose_dir = r"co_dat_only2per/openpose"
depth_pure_background_dir = r"co_dat_only2per/depth_pure_background"
depth_original_image_dir = r"co_dat_only2per/depth_original_image"
mask_dir = r"co_dat_only2per/mask"

# 获取所有原始图文件名
raw_files = [os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 检查 OpenPose
openpose_files = [os.path.splitext(f)[0] for f in os.listdir(openpose_dir)]
# 检查 Depth
depth_pure_background_files = [os.path.splitext(f)[0] for f in os.listdir(depth_pure_background_dir)]
depth_original_image_files = [os.path.splitext(f)[0] for f in os.listdir(depth_original_image_dir)]
# 检查 Mask
mask_files = [os.path.splitext(f)[0] for f in os.listdir(mask_dir)]

# 统计匹配情况
match_count = 0
for f in raw_files:
    if f in openpose_files and f in depth_pure_background_files and f in depth_original_image_files and f in mask_files:
        match_count += 1
    else:
        print(f"Err Missing control chart:{f}")

print(f"\n Number of fully matched samples: {match_count}/{len(raw_files)}")
print(f" OpenPose sample count: {len(openpose_files)}")
print(f" Depth_pure_background sample count: {len(depth_pure_background_files)}")
print(f" Depth_original_image sample count: {len(depth_original_image_files)}")
print(f" Mask sample count: {len(mask_files)}")