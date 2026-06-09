import os
import cv2
import numpy as np
import json
import itertools  # 用于自动生成两两角色组合

# ===================== 全局配置 =====================
OUTPUT_ROOT = "synthetic_test_dataset"
SCENES = ["side_by_side", "handshake", "front_back"]
IMAGE_SIZE = 512

# 统一的 OpenPose 骨骼简化连接线
CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 7), (7, 8), (8, 9), (8, 10)]

# ===================== 配置：5个下载的 LoRA 角色模型信息 =====================
# 精确匹配文件名与 Civitai 触发词
CHARACTERS = [
    {
        "id": "Asuna",
        "lora_name": "asuna_(stacia)-v1.5",
        "triggers": "stacia, white dress, armor, white thighhighs"
    },
    {
        "id": "YamasakiAnzu",
        "lora_name": "gantzyamasakianzu",
        "triggers": "yamasaki_anzu, bodysuit"
    },
    {
        "id": "Vanilla",
        "lora_name": "super-vanilla-newlora-ver1-p",
        "triggers": "vanilla, xiangcao, shouban, girls"
    },
    {
        "id": "TogaHimiko",
        "lora_name": "TogaHimiko-01",
        "triggers": "Toga, Himiko, Himiko Toga"
    },
    {
        "id": "OchacoUraraka",
        "lora_name": "OchacoUraraka-01",
        "triggers": "Ochaco, Bodysuit, Hero suit, white shirt, red necktie, green skirt, U.A School Uniform"
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
    """两人并排站立，略微偏左右"""
    kps_p1 = np.array([
        [0.30, 0.50],  # 鼻子
        [0.28, 0.58], [0.32, 0.58],  # 左右肩
        [0.26, 0.68], [0.34, 0.68],  # 左右肘
        [0.24, 0.78], [0.36, 0.78],  # 左右腕
        [0.30, 0.60], [0.30, 0.70],  # 髋
        [0.28, 0.80], [0.32, 0.80]   # 膝
    ])
    kps_p2 = np.array([
        [0.70, 0.50],
        [0.68, 0.58], [0.72, 0.58],
        [0.66, 0.68], [0.74, 0.68],
        [0.64, 0.78], [0.76, 0.78],
        [0.70, 0.60], [0.70, 0.70],
        [0.68, 0.80], [0.72, 0.80]
    ])
    all_kps = np.vstack([kps_p1, kps_p2])
    return draw_pose_keypoints(all_kps, CONNECTIONS, IMAGE_SIZE)

def generate_pose_handshake():
    """两人面对面，手部相交"""
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
    """一前一后，前面的人物稍大（y 坐标稍大）"""
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
    """生成有空间感的深度图（0-255）"""
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    # 背景渐变：上深下浅
    for i in range(IMAGE_SIZE):
        depth[i, :] = int(100 + (i / IMAGE_SIZE) * 100)
    if scene == "front_back":
        # 前方人物区域更近（深度值更大）
        cv2.ellipse(depth, (IMAGE_SIZE//2, IMAGE_SIZE//2+30), (120, 180), 0, 0, 360, 200, -1)
        cv2.ellipse(depth, (IMAGE_SIZE//2, IMAGE_SIZE//2-60), (100, 150), 0, 0, 360, 130, -1)
    else:
        # 左右两个人物区域
        cv2.ellipse(depth, (IMAGE_SIZE//4, IMAGE_SIZE//2), (80, 180), 0, 0, 360, 200, -1)
        cv2.ellipse(depth, (3*IMAGE_SIZE//4, IMAGE_SIZE//2), (80, 180), 0, 0, 360, 200, -1)
    
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

def generate_mask(scene):
    """生成人物实例掩码（角色1像素值为1，角色2像素值为2）"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if scene == "side_by_side":
        # 左人物椭圆 ID=1
        cv2.ellipse(mask, (IMAGE_SIZE//4, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 128, -1)
        # 右人物椭圆 ID=2
        cv2.ellipse(mask, (3*IMAGE_SIZE//4, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 255, -1)
    elif scene == "handshake":
        # 两人靠近，椭圆有重叠
        cv2.ellipse(mask, (IMAGE_SIZE//3, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 128, -1)
        cv2.ellipse(mask, (2*IMAGE_SIZE//3, IMAGE_SIZE//2), (70, 180), 0, 0, 360, 255, -1)
    elif scene == "front_back":
        # 后面人物较小 ID=2，前面人物较大 ID=1
        cv2.ellipse(mask, (IMAGE_SIZE//2, IMAGE_SIZE//2-50), (80, 120), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (IMAGE_SIZE//2, IMAGE_SIZE//2+40), (90, 160), 0, 0, 360, 128, -1)
    return mask

def generate_prompt_config(scene, char1, char2):
    """
    根据配对的两个角色动态生成标准多角色 Prompt
    同时封装 LoRA 信息，支持解析或直接在 WebUI/Diffusers 中载入
    """
    p1_tags = char1["triggers"]
    p2_tags = char2["triggers"]
    
    # 构建包含显式角色标识的多角色标准推理 Prompt
    prompt = f"person1: a photo of {p1_tags}, person2: a photo of {p2_tags}, {scene.replace('_', ' ')} scene, high quality, 8k, realistic"
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

# ===================== 主流程：全量组合生成 =====================
def main():
    print(" 开始生成结构化多角色合成实验测试集...")
    
    # 创建标准的检测输入目录
    for folder in ["poses", "depths", "masks"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

    all_configs = []
    
    # 使用 itertools.combinations 进行 5 选 2 的角色组合，不重复地生成所有对比对（共 10 种独立角色配置）
    character_pairs = list(itertools.combinations(CHARACTERS, 2))

    for scene in SCENES:
        # 1. 根据当前交互场景生成对应的基础布局控制图
        if scene == "side_by_side":
            pose = generate_pose_side_by_side()
        elif scene == "handshake":
            pose = generate_pose_handshake()
        elif scene == "front_back":
            pose = generate_pose_front_back()
            
        depth = generate_depth(scene)
        mask = generate_mask(scene)

        # 2. 遍历所有角色组合，并绑定至当前的布局空间中
        for char1, char2 in character_pairs:
            # 构造样本的唯一标识符（例如: side_by_side_Asuna_vs_YamasakiAnzu）
            sample_id = f"{scene}_{char1['id']}_vs_{char2['id']}"
            
            # 生成带特征标签的 prompt 配置
            config = generate_prompt_config(scene, char1, char2)
            config["sample_id"] = sample_id
            
            # 3. 将图像阵列以 sample_id 为名保存
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "poses", f"{sample_id}.png"), pose)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "depths", f"{sample_id}.png"), depth)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "masks", f"{sample_id}.png"), mask)
            
            all_configs.append(config)
            print(f"  [+] 已规划样本控制节点: {sample_id}")

    # 4. 导出集成式的清单配置文件，供模型推理和评测脚本一键载入
    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
        
    print(f"\n 精确合成测试集已构建成功！")
    print(f"总计样本规模: {len(all_configs)} 组 (3 种复杂场景 × 10 类跨人物配对)")
    print(f"清单文件已保存至: {os.path.abspath(config_output_path)}")

if __name__ == "__main__":
    main()