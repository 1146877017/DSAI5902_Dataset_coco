import os
import cv2
import numpy as np
import json
import itertools

# ===================== 全局配置 =====================
OUTPUT_ROOT = "synthetic_test_dataset"
SCENES = ["side_by_side", "handshake", "front_back"]
BACKGROUNDS = ["outdoor_park", "indoor_room", "futuristic_street"] # 显式引入 3 种背景环境 
IMAGE_SIZE = 512

# 统一几何参数配置
ELLIPSE_SIZES = {
    "side_by_side": (75, 180),
    "handshake": (75, 180),
    "front_back_back": (85, 130),
    "front_back_front": (110, 170)
}

# 统一的 OpenPose 骨骼简化连接线
CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 7), (7, 8), (8, 9), (8, 10)]

# ===================== 配置： 4 个 LoRA 角色模型信息 =====================
CHARACTERS = [
    {
        "id": "Asuna",
        "lora_name": "asuna_(stacia)-v1.5",
        "triggers": "stacia, white dress, armor, white thighhighs"
    },
    {
        "id": "Neferpitou",
        "lora_name": "LoRA_Neferpitou",
        "triggers": "NeferpitouDef, white hair, wavy hair, cat ears, cat tail, blue shirt, joints, doll joints"
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
def draw_pose_keypoints(keypoints_list, connections, img_size=512):
    """绘制 OpenPose 风格姿态图，完美支持多人物骨骼独立连线"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    
    for keypoints in keypoints_list:
        for kp in keypoints:
            x, y = int(kp[0] * w), int(kp[1] * h)
            cv2.circle(img, (x, y), 4, (0, 255, 255), -1)
        for (start, end) in connections:
            x1, y1 = int(keypoints[start][0] * w), int(keypoints[start][1] * h)
            x2, y2 = int(keypoints[end][0] * w), int(keypoints[end][1] * h)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return img

def generate_pose_side_by_side():
    # 人物中心横坐标精确对应 0.30 与 0.70 
    kps_p1 = np.array([[0.30, 0.50], [0.28, 0.58], [0.32, 0.58], [0.26, 0.68], [0.34, 0.68], [0.24, 0.78], [0.36, 0.78], [0.30, 0.60], [0.30, 0.70], [0.28, 0.80], [0.32, 0.80]])
    kps_p2 = np.array([[0.70, 0.50], [0.68, 0.58], [0.72, 0.58], [0.66, 0.68], [0.74, 0.68], [0.64, 0.78], [0.76, 0.78], [0.70, 0.60], [0.70, 0.70], [0.68, 0.80], [0.72, 0.80]])
    return draw_pose_keypoints([kps_p1, kps_p2], CONNECTIONS, IMAGE_SIZE)

def generate_pose_handshake():
    # 人物中心横坐标精确对应 0.35 与 0.65
    kps_p1 = np.array([[0.35, 0.50], [0.32, 0.55], [0.38, 0.55], [0.30, 0.65], [0.40, 0.65], [0.28, 0.75], [0.42, 0.75], [0.35, 0.60], [0.35, 0.70], [0.33, 0.80], [0.37, 0.80]])
    kps_p2 = np.array([[0.65, 0.50], [0.62, 0.55], [0.68, 0.55], [0.60, 0.65], [0.70, 0.65], [0.58, 0.75], [0.72, 0.75], [0.65, 0.60], [0.65, 0.70], [0.63, 0.80], [0.67, 0.80]])
    return draw_pose_keypoints([kps_p1, kps_p2], CONNECTIONS, IMAGE_SIZE)

def generate_pose_front_back():
    # 后景纵向质心约 0.61，前景纵向质心约 0.71
    kps_front = np.array([[0.50, 0.55], [0.46, 0.62], [0.54, 0.62], [0.42, 0.72], [0.58, 0.72], [0.40, 0.82], [0.60, 0.82], [0.50, 0.67], [0.50, 0.77], [0.48, 0.87], [0.52, 0.87]])
    kps_back = np.array([[0.50, 0.45], [0.46, 0.52], [0.54, 0.52], [0.42, 0.62], [0.58, 0.62], [0.40, 0.72], [0.60, 0.72], [0.50, 0.57], [0.50, 0.67], [0.48, 0.77], [0.52, 0.77]])
    return draw_pose_keypoints([kps_front, kps_back], CONNECTIONS, IMAGE_SIZE)

def generate_depth(scene, bg):
    """生成具备明确空间遮挡的渐变深度图，包含背景结构以完整覆盖至少 3 种背景要求 """
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    # 基础远景渐变
    for i in range(IMAGE_SIZE):
        depth[i, :] = int(50 + (i / IMAGE_SIZE) * 50)
    
    # 绘制背景的控制线几何结构 
    if bg == "indoor_room":
        cv2.line(depth, (0, 0), (120, 150), 80, 2)
        cv2.line(depth, (IMAGE_SIZE, 0), (IMAGE_SIZE-120, 150), 80, 2)
        cv2.line(depth, (120, 150), (IMAGE_SIZE-120, 150), 70, 2)
    elif bg == "outdoor_park":
        pts = np.array([[0, 250], [100, 180], [250, 230], [400, 150], [IMAGE_SIZE, 220], [IMAGE_SIZE, 300], [0, 300]], np.int32)
        cv2.fillPoly(depth, [pts], 75)
    elif bg == "futuristic_street":
        cv2.rectangle(depth, (0, 0), (80, 400), 65, -1)
        cv2.rectangle(depth, (IMAGE_SIZE-80, 0), (IMAGE_SIZE, 400), 65, -1)
    
    # 绘制与骨骼图及掩码图“绝对像素级对齐”的人物实体深度
    if scene == "side_by_side":
        c1, c2 = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        v_center = int(0.65 * IMAGE_SIZE)
        cv2.ellipse(depth, (c1, v_center), ELLIPSE_SIZES["side_by_side"], 0, 0, 360, 180, -1)
        cv2.ellipse(depth, (c2, v_center), ELLIPSE_SIZES["side_by_side"], 0, 0, 360, 180, -1)
    elif scene == "handshake":
        c1, c2 = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        v_center = int(0.65 * IMAGE_SIZE)
        cv2.ellipse(depth, (c1, v_center), ELLIPSE_SIZES["handshake"], 0, 0, 360, 180, -1)
        cv2.ellipse(depth, (c2, v_center), ELLIPSE_SIZES["handshake"], 0, 0, 360, 180, -1)
    elif scene == "front_back":
        v_back = int(0.61 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        # 后景人物：灰度130（较远）
        cv2.ellipse(depth, (IMAGE_SIZE//2, v_back), ELLIPSE_SIZES["front_back_back"], 0, 0, 360, 130, -1)
        # 前景人物：灰度210（较近，完美确立空间深度层级）
        cv2.ellipse(depth, (IMAGE_SIZE//2, v_front), ELLIPSE_SIZES["front_back_front"], 0, 0, 360, 210, -1)
                
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

def generate_mask(scene):
    """生成精确对应的实例掩码图（与姿态、深度图空间坐标绝对对齐）"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if scene == "side_by_side":
        c1, c2 = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        v_center = int(0.65 * IMAGE_SIZE)
        cv2.ellipse(mask, (c1, v_center), ELLIPSE_SIZES["side_by_side"], 0, 0, 360, 128, -1) # person1 -> 128
        cv2.ellipse(mask, (c2, v_center), ELLIPSE_SIZES["side_by_side"], 0, 0, 360, 255, -1) # person2 -> 255
    elif scene == "handshake":
        c1, c2 = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        v_center = int(0.65 * IMAGE_SIZE)
        cv2.ellipse(mask, (c1, v_center), ELLIPSE_SIZES["handshake"], 0, 0, 360, 128, -1)
        cv2.ellipse(mask, (c2, v_center), ELLIPSE_SIZES["handshake"], 0, 0, 360, 255, -1)
    elif scene == "front_back":
        v_back = int(0.61 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        # 先画远景标签（person2 -> 255），再用前景标签覆盖它（person1 -> 128）
        cv2.ellipse(mask, (IMAGE_SIZE//2, v_back), ELLIPSE_SIZES["front_back_back"], 0, 0, 360, 255, -1) 
        cv2.ellipse(mask, (IMAGE_SIZE//2, v_front), ELLIPSE_SIZES["front_back_front"], 0, 0, 360, 128, -1) 
    return mask

def generate_prompt_config(scene, bg, char1, char2):
    p1_tags = char1["triggers"]
    p2_tags = char2["triggers"]
    prompt = f"person1: a photo of {p1_tags}. person2: a photo of {p2_tags}. {scene.replace('_', ' ')} scene in a {bg.replace('_', ' ')}, high quality, 8k, realistic"
    neg_prompt = "blurry, low quality, distorted, missing people, extra limbs, monochrome"
    return {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "scene": scene,
        "background": bg,
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
    character_pairs = list(itertools.permutations(CHARACTERS, 2))

    # 使用 itertools.product 替代 zip，实现 3 种互动动作与 3 种背景环境的完全交叉覆盖 (共 9 种环境模态组合)
    for scene, bg in itertools.product(SCENES, BACKGROUNDS):
        # 根据当前场景单独产生对应的基础控制特征阵列
        if scene == "side_by_side":
            pose = generate_pose_side_by_side()
        elif scene == "handshake":
            pose = generate_pose_handshake()
        else:
            pose = generate_pose_front_back()
            
        depth = generate_depth(scene, bg)
        mask = generate_mask(scene)

        for char1, char2 in character_pairs:
            sample_id = f"{scene}_{bg}_{char1['id']}_vs_{char2['id']}"
            config = generate_prompt_config(scene, bg, char1, char2)
            config["sample_id"] = sample_id
            
            # 保持 1:1 的数据配对落盘，使下游测试框架能够直接根据 sample_id 索引多模态输入文件
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "poses", f"{sample_id}.png"), pose)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "depths", f"{sample_id}.png"), depth)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "masks", f"{sample_id}.png"), mask)
            all_configs.append(config)
            print(f"  [+] 已规划样本控制节点: {sample_id}")

    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
        
    print(f"\n 精确合成测试集已构建成功！")
    print(f"总计样本规模: {len(all_configs)} 组 (3 种互动场景 × 3 种物理背景 × 12 类跨角色空间排列组合 = 108组)")

if __name__ == "__main__":
    main()