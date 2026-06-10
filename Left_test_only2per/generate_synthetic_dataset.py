import os
import cv2
import numpy as np
import json
import itertools

# ===================== 全局配置 =====================
OUTPUT_ROOT = "synthetic_test_dataset"
SCENES = ["side_by_side", "handshake", "front_back"]
IMAGE_SIZE = 512

# 统一的 OpenPose 骨骼简化连接线
CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 7), (7, 8), (8, 9), (8, 10)]

# ===================== 配置： LoRA 角色/风格模型信息 =====================
CHARACTERS = [
    {
        "id": "Asuna",
        "lora_name": "asuna_(stacia)-v1.5",
        "triggers": "stacia, white dress, armor, white thighhighs"
    },
    {
        "id": "CuteRichStyle",   # 保持映射
        "lora_name": "dicuki",
        "triggers": "cbzbb style, intricate details, vibrant colors"   
    },
    {
        "id": "TogaHimiko",
        "lora_name": "TogaHimiko-01",
        "triggers": "Toga, Himiko, Himiko Toga"
    },
    {
        "id": "OchacoUraraka",
        "lora_name": "OchacoUraraka-01",
        "triggers": "Ochaco, Bodysuit, Hero suit, white shirt, red necktie, green skirt"
    }
]

# ===================== 辅助函数：图像控制块生成 =====================
def draw_pose_keypoints(keypoints, connections, img_size=512):
    """绘制 OpenPose 风格姿态图"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    for kp in keypoints:
        x, y = int(kp[0] * w), int(kp[1] * h)
        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)
    for (start, end) in connections:
        x1, y1 = int(keypoints[start][0] * w), int(keypoints[start][1] * h)
        x2, y2 = int(keypoints[end][0] * w), int(keypoints[end][1] * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return img

def generate_pose_side_by_side():
    """两人并排站立，定位精准"""
    kps_p1 = np.array([
        [0.30, 0.50], [0.28, 0.58], [0.32, 0.58], [0.26, 0.68], [0.34, 0.68],
        [0.24, 0.78], [0.36, 0.78], [0.30, 0.60], [0.30, 0.70], [0.28, 0.80], [0.32, 0.80]
    ])
    kps_p2 = np.array([
        [0.70, 0.50], [0.68, 0.58], [0.72, 0.58], [0.66, 0.68], [0.74, 0.68],
        [0.64, 0.78], [0.76, 0.78], [0.70, 0.60], [0.70, 0.70], [0.68, 0.80], [0.72, 0.80]
    ])
    all_kps = np.vstack([kps_p1, kps_p2])
    return draw_pose_keypoints(all_kps, CONNECTIONS, IMAGE_SIZE)

def generate_pose_handshake():
    """两人面对面手部交叉"""
    kps_p1 = np.array([
        [0.35, 0.50], [0.32, 0.55], [0.38, 0.55], [0.30, 0.65], [0.40, 0.65],
        [0.28, 0.75], [0.42, 0.75], [0.35, 0.60], [0.35, 0.70], [0.33, 0.80], [0.37, 0.80]
    ])
    kps_p2 = np.array([
        [0.65, 0.50], [0.62, 0.55], [0.68, 0.55], [0.60, 0.65], [0.70, 0.65],
        [0.58, 0.75], [0.72, 0.75], [0.65, 0.60], [0.65, 0.70], [0.63, 0.80], [0.67, 0.80]
    ])
    all_kps = np.vstack([kps_p1, kps_p2])
    return draw_pose_keypoints(all_kps, CONNECTIONS, IMAGE_SIZE)

def generate_pose_front_back():
    """一前一后布局"""
    kps_front = np.array([
        [0.50, 0.55], [0.46, 0.62], [0.54, 0.62], [0.42, 0.72], [0.58, 0.72],
        [0.40, 0.82], [0.60, 0.82], [0.50, 0.67], [0.50, 0.77], [0.48, 0.87], [0.52, 0.87]
    ])
    kps_back = np.array([
        [0.50, 0.45], [0.46, 0.52], [0.54, 0.52], [0.42, 0.62], [0.58, 0.62],
        [0.40, 0.72], [0.60, 0.72], [0.50, 0.57], [0.50, 0.67], [0.48, 0.77], [0.52, 0.77]
    ])
    all_kps = np.vstack([kps_front, kps_back])
    return draw_pose_keypoints(all_kps, CONNECTIONS, IMAGE_SIZE)

def generate_depth(scene):
    """生成具备明确空间遮挡的渐变深度图"""
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(IMAGE_SIZE):
        depth[i, :] = int(100 + (i / IMAGE_SIZE) * 100)
    if scene == "front_back":
        # 后面人物（ID=2）较远，亮度较低；前面人物（ID=1）更近，亮度较高
        cv2.ellipse(depth, (IMAGE_SIZE//2, IMAGE_SIZE//2-60), (100, 150), 0, 0, 360, 130, -1)
        cv2.ellipse(depth, (IMAGE_SIZE//2, IMAGE_SIZE//2+30), (120, 180), 0, 0, 360, 200, -1)
    else:
        cv2.ellipse(depth, (IMAGE_SIZE//4, IMAGE_SIZE//2), (80, 180), 0, 0, 360, 200, -1)
        cv2.ellipse(depth, (3*IMAGE_SIZE//4, IMAGE_SIZE//2), (80, 180), 0, 0, 360, 200, -1)
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

def generate_mask(scene):
    """生成精确对应的实例掩码图"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if scene == "side_by_side":
        cv2.ellipse(mask, (IMAGE_SIZE//4, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 128, -1) # person1 -> 128
        cv2.ellipse(mask, (3*IMAGE_SIZE//4, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 255, -1) # person2 -> 255
    elif scene == "handshake":
        cv2.ellipse(mask, (IMAGE_SIZE//3, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 128, -1)
        cv2.ellipse(mask, (2*IMAGE_SIZE//3, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 255, -1)
    elif scene == "front_back":
        # 先画后景人物（person2），再用前景人物（person1）进行覆盖覆盖，建立完美遮挡逻辑
        cv2.ellipse(mask, (IMAGE_SIZE//2, IMAGE_SIZE//2-50), (80, 120), 0, 0, 360, 255, -1) # person2
        cv2.ellipse(mask, (IMAGE_SIZE//2, IMAGE_SIZE//2+40), (90, 160), 0, 0, 360, 128, -1) # person1
    return mask

def generate_prompt_config(scene, char1, char2):
    p1_tags = char1["triggers"]
    p2_tags = char2["triggers"]
    # 建立严格契合分词器检索模式的硬性声明 Prompt 结构
    prompt = f"person1: a photo of {p1_tags}. person2: a photo of {p2_tags}. {scene.replace('_', ' ')} scene, high quality, 8k, realistic"
    neg_prompt = "blurry, low quality, distorted, missing people, extra limbs, monochrome"
    return {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "scene": scene,
        "characters": [char1["id"], char2["id"]],
        "lora_details": [
            {"character_id": char1["id"], "lora_name": char1["lora_name"], "weight": 0.8},
            {"character_id": char2["id"], "lora_name": char2["lora_name"], "weight": 0.8}
        ]
    }

def main():
    print(" 开始生成结构化多角色合成实验测试集...")
    for folder in ["poses", "depths", "masks"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

    all_configs = []
    # 改用 permutations，使角色在（左/右、前/后）物理站位上均有样本对称分布
    character_pairs = list(itertools.permutations(CHARACTERS, 2))

    for scene in SCENES:
        pose = generate_pose_side_by_side() if scene == "side_by_side" else (
            generate_pose_handshake() if scene == "handshake" else generate_pose_front_back()
        )
        depth = generate_depth(scene)
        mask = generate_mask(scene)

        for char1, char2 in character_pairs:
            sample_id = f"{scene}_{char1['id']}_vs_{char2['id']}"
            config = generate_prompt_config(scene, char1, char2)
            config["sample_id"] = sample_id
            
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "poses", f"{sample_id}.png"), pose)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "depths", f"{sample_id}.png"), depth)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "masks", f"{sample_id}.png"), mask)
            all_configs.append(config)
            print(f"  [+] 已规划样本控制节点: {sample_id}")

    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
    print(f"\n 精确合成测试集已构建成功！")
    print(f"总计样本规模: {len(all_configs)} 组 (3 种复杂场景 × 12 类跨人物空间排列组合)")

if __name__ == "__main__":
    main()