import os

# 路径配置
raw_dir = r"coco_multi_person/raw"
openpose_dir = r"coco_multi_person/openpose"
depth_dir = r"coco_multi_person/depth"
mask_dir = r"coco_multi_person/mask"

# 获取所有原始图文件名
raw_files = [os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 检查 OpenPose
openpose_files = [os.path.splitext(f)[0] for f in os.listdir(openpose_dir)]
# 检查 Depth
depth_files = [os.path.splitext(f)[0] for f in os.listdir(depth_dir)]
# 检查 Mask
mask_files = [os.path.splitext(f)[0] for f in os.listdir(mask_dir)]

# 统计匹配情况
match_count = 0
for f in raw_files:
    if f in openpose_files and f in depth_files and f in mask_files:
        match_count += 1
    else:
        print(f"⚠️ 缺失控制图: {f}")

print(f"\n✅ 完整匹配的样本数: {match_count}/{len(raw_files)}")
print(f"✅ OpenPose 样本数: {len(openpose_files)}")
print(f"✅ Depth 样本数: {len(depth_files)}")
print(f"✅ Mask 样本数: {len(mask_files)}")