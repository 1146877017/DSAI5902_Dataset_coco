import os
import cv2
import numpy as np
import json

# ===================== 配置 : 合成数据集 =====================
# 输出路径
OUTPUT_ROOT = "synthetic_test_dataset"
# 3种核心双人交互场景
SCENES = ["side_by_side", "handshake", "front_back"]
# 图片尺寸
IMAGE_SIZE = 512
# 4个固定LoRA角色
CHARACTERS = [
    "red_hair_girl",
    "black_hair_boy",
    "cat_ear_girl",
    "white_beard_old_man"
]

# ===================== 创建文件夹 =====================
def build_folders():
    folders = [
        "poses",      # OpenPose姿态图
        "depths",     # 深度图
        "masks",      # 人物实例掩码
        "prompts"     # 场景prompt配置
    ]
    for f in folders:
        os.makedirs(os.path.join(OUTPUT_ROOT, f), exist_ok=True)
    print(" 合成数据集文件夹创建完成")

# ===================== 生成标准双人姿态图 : OpenPose格式 =====================
def generate_pose(scene):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    h, w = IMAGE_SIZE, IMAGE_SIZE

    # 3种场景的骨骼坐标
    if scene == "side_by_side":
        # 人物1 左
        cv2.circle(img, (w//4, h//2), 5, (0,255,255), -1)
        cv2.line(img, (w//4, h//2), (w//4, h//2+80), (0,255,255), 2)
        # 人物2 右
        cv2.circle(img, (3*w//4, h//2), 5, (0,255,255), -1)
        cv2.line(img, (3*w//4, h//2), (3*w//4, h//2+80), (0,255,255), 2)

    elif scene == "handshake":
        # 握手姿态
        cv2.circle(img, (w//3, h//2), 5, (0,255,255), -1)
        cv2.circle(img, (2*w//3, h//2), 5, (0,255,255), -1)
        cv2.line(img, (w//3, h//2), (w//3+60, h//2), (0,255,255), 2)
        cv2.line(img, (2*w//3, h//2), (2*w//3-60, h//2), (0,255,255), 2)

    elif scene == "front_back":
        # 前后遮挡姿态
        cv2.circle(img, (w//2, h//2), 5, (0,255,255), -1)
        cv2.circle(img, (w//2, h//2-60), 5, (0,255,255), -1)

    return img

# ===================== 生成深度图 : 渐变深度 =====================
def generate_depth(scene):
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if scene == "front_back":
        # 前后场景：前景深，背景浅
        depth[:, :] = np.linspace(100, 200, IMAGE_SIZE)
    else:
        # 均匀深度
        depth[:, :] = 150
    return cv2.applyColorMap(depth, cv2.COLORMAP_JET)

# ===================== 生成实例掩码 : 2个人物 =====================
def generate_mask(scene):
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if scene in ["side_by_side", "handshake"]:
        # 左右两人：ID=1, ID=2
        mask[:, :IMAGE_SIZE//2] = 1
        mask[:, IMAGE_SIZE//2:] = 2
    elif scene == "front_back":
        # 前后两人：ID=1(前), ID=2(后)
        mask[:IMAGE_SIZE//2, :] = 2
        mask[IMAGE_SIZE//2:, :] = 1
    return mask

# ===================== 生成标准Prompt =====================
def generate_prompt(scene):
    prompt = f"""
person1: a photo of {CHARACTERS[0]},
person2: a photo of {CHARACTERS[1]},
{scene.replace('_', ' ')} scene, high quality, 8k, realistic
    """.strip().replace("\n", " ")
    
    neg_prompt = "blurry, low quality, distorted, missing people, extra limbs"
    
    return {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "scene": scene,
        "characters": [CHARACTERS[0], CHARACTERS[1]]
    }

# ===================== 生成流程 =====================
def main():
    print(" 开始生成 Proposal 合成测试集...")
    build_folders()
    all_configs = []

    for scene in SCENES:
        # 生成文件
        pose = generate_pose(scene)
        depth = generate_depth(scene)
        mask = generate_mask(scene)
        prompt = generate_prompt(scene)

        # 保存路径
        base_path = os.path.join(OUTPUT_ROOT)
        cv2.imwrite(os.path.join(base_path, "poses", f"{scene}.jpg"), pose)
        cv2.imwrite(os.path.join(base_path, "depths", f"{scene}.jpg"), depth)
        cv2.imwrite(os.path.join(base_path, "masks", f"{scene}.png"), mask)
        
        # 保存配置
        all_configs.append(prompt)
        print(f" 生成完成：{scene}")

    # 保存总配置
    with open(os.path.join(OUTPUT_ROOT, "synthetic_configs.json"), "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)

    print("\n 合成测试集全部生成完成！")
    print(f" 路径：{OUTPUT_ROOT}")
    print(f"包含：3个场景 + 姿态图/深度图/掩码图/prompt配置")

if __name__ == "__main__":
    main()