import cv2
import numpy as np

# 检查掩码是否正确分离
# mask = cv2.imread("synthetic_test_dataset/masks/side_by_side_outdoor_park_Sera_vs_TogaHimiko.png", cv2.IMREAD_UNCHANGED)
# print("Alpha channel unique values:", np.unique(mask[:, :, 3]))
# 修复后应输出: [128 255]

# 这个报错直接揭示了特征隔离失效的真正元凶！
 # 报错原因分析
# 报错 array is 2-dimensional 说明 cv2.imread(..., cv2.IMREAD_UNCHANGED) 读出来的掩码图只有 2 个维度（单通道灰度图），根本没有第 4 个 Alpha 通道！
# 回顾 generate_synthetic_dataset.py：



# mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) # 创建的是 2D 单通道图
# cv2.ellipse(mask, ..., 128, -1) # 直接画灰度值 128
# cv2.imwrite(..., mask)          # 保存为单通道 PNG
# 结论：掩码图是单通道灰度图（像素值直接是 0, 128, 255），而不是带透明度的 RGBA 图！
 # 为什么之前的 Method 两人风格一模一样？
# 在之前的 process_mask 中：

# alpha = mask[:, :, 3] if mask.shape[-1] == 4 else np.ones_like(mask) * 255
# 因为图是 2D 的，mask.shape[-1] 是 512（不等于 4），所以代码走到了 else 分支，把 alpha 强行设成了全 255！
# 结果：
# mask1 (Sera) = (255 == 128) = 全黑 (0) ❌ (Sera 的掩码完全丢失！)
# mask2 (Toga) = (255 == 255) = 全白 (255) ✅
# 因为 Sera 没有掩码保护，她的特征被全局污染，或者两人的特征在计算时发生了严重的混淆，导致最终生成的两人风格完全一致。

# 修复
mask_path = "synthetic_test_dataset/masks/side_by_side_outdoor_park_Sera_vs_TogaHimiko.png"
# 以灰度图读取
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Error: Could not read mask file.")
else:
    unique_vals = np.unique(mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Unique pixel values: {unique_vals}")
    
    # 统计各区域像素占比
    total_pixels = mask.size
    p1_ratio = np.sum(mask == 128) / total_pixels * 100
    p2_ratio = np.sum(mask == 255) / total_pixels * 100
    
    print(f"Person 1 (128) area: {p1_ratio:.2f}%")
    print(f"Person 2 (255) area: {p2_ratio:.2f}%")
    
    
    
# 关键结论：
# 掩码格式正确：确实是单通道灰度图（非RGBA），包含0, 128, 255三个值
# 角色分离精准：Sera (128) 占12.05%，Toga (255) 占11.25% → 完美对称分布
# 问题根源确认：之前的 process_mask 函数错误地尝试读取Alpha通道，导致掩码处理失效
 # 为什么之前 method 图中两人风格一模一样？
# 在原始 process_mask 函数中：

# alpha = mask[:, :, 3] if mask.shape[-1] == 4 else np.ones_like(mask) * 255
# 因为掩码是单通道灰度图（shape=(512,512)），所以 mask.shape[-1] 是 512（≠4），导致：
# Sera 的掩码 = (255 == 128) → 全0（黑色） → 完全丢失！
# Toga 的掩码 = (255 == 255) → 全255（白色） → 完全覆盖
# 结果：Sera 区域没有任何掩码保护，她的特征被 Toga 的 LoRA 污染 → 两人风格完全一致