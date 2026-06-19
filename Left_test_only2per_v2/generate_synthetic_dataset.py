import os
import cv2
import numpy as np
import json
import itertools

# ===================== 全局配置 =====================
OUTPUT_ROOT = "synthetic_test_dataset"
SCENES = ["side_by_side", "handshake", "front_back"]
BACKGROUNDS = ["outdoor_park", "indoor_room", "futuristic_street"] 
IMAGE_SIZE = 512

# 将所有场景的角色尺寸全部解耦对齐，采用 front_back 风格的显式命名体系
ELLIPSE_SIZES = {
    "side_by_side_p1": (75, 180),
    "side_by_side_p2": (75, 180),
    "handshake_p1": (75, 180),
    "handshake_p2": (75, 180),
    "front_back_back": (85, 130),
    "front_back_front": (110, 170)
}

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

# ===================== 统一的场景坐标映射器 =====================
def get_scene_keypoints(scene):
    """统一管理场景关键点，确保 Pose 控制与 Mask 肢体扩张完全同步"""
    if scene == "side_by_side":
        kps_p1 = np.array([[0.30, 0.50], [0.28, 0.58], [0.32, 0.58], [0.26, 0.68], [0.34, 0.68], [0.24, 0.78], [0.36, 0.78], [0.30, 0.60], [0.30, 0.70], [0.28, 0.80], [0.32, 0.80]])
        kps_p2 = np.array([[0.70, 0.50], [0.68, 0.58], [0.72, 0.58], [0.66, 0.68], [0.74, 0.68], [0.64, 0.78], [0.76, 0.78], [0.70, 0.60], [0.70, 0.70], [0.68, 0.80], [0.72, 0.80]])
    elif scene == "handshake":
        kps_p1 = np.array([[0.35, 0.48], [0.30, 0.55], [0.40, 0.55], [0.27, 0.65], [0.45, 0.60], [0.25, 0.75], [0.50, 0.62], [0.35, 0.60], [0.35, 0.72], [0.32, 0.85], [0.38, 0.85]])
        kps_p2 = np.array([[0.65, 0.48], [0.60, 0.55], [0.70, 0.55], [0.55, 0.60], [0.73, 0.65], [0.50, 0.62], [0.75, 0.75], [0.65, 0.60], [0.65, 0.72], [0.62, 0.85], [0.67, 0.85]])
    else:  # front_back
        kps_p1 = np.array([
            [0.44, 0.42], [0.38, 0.52], [0.50, 0.52], [0.34, 0.65], [0.54, 0.65], 
            [0.32, 0.78], [0.56, 0.78], [0.44, 0.49], [0.44, 0.72], [0.40, 0.88], [0.48, 0.88]
        ])
        kps_p2 = np.array([
            [0.58, 0.35], [0.54, 0.42], [0.62, 0.42], [0.51, 0.52], [0.65, 0.52], 
            [0.49, 0.62], [0.67, 0.62], [0.58, 0.40], [0.58, 0.58], [0.55, 0.72], [0.61, 0.72]
        ])
    return [kps_p1, kps_p2]

# ===================== 标准多色 OpenPose 骨骼图生成 =====================
def draw_pose_keypoints(keypoints_list, img_size=512):
    """绘制完美兼容 lllyasviel/control_v11p_sd15_openpose 标准的 COCO 色调彩色骨骼图"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    
    COCO_PAIRS = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
    ]
    LINE_COLORS = [
        [0, 0, 255], [255, 0, 0], [0, 85, 255], [0, 170, 255], [0, 255, 255], [0, 255, 170],
        [0, 255, 85], [0, 255, 0], [85, 255, 0], [170, 255, 0], [255, 255, 0], [255, 170, 0],
        [255, 85, 0], [255, 0, 0], [255, 0, 85], [255, 0, 170], [255, 0, 255]
    ]
    JOINT_COLORS = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
    ]

    for user_kps in keypoints_list:
        coco_kps = np.zeros((18, 2))
        coco_kps[0] = user_kps[0]   # Nose
        coco_kps[1] = user_kps[7]   # Neck
        coco_kps[2] = user_kps[2]   # RShoulder
        coco_kps[3] = user_kps[4]   # RElbow
        coco_kps[4] = user_kps[6]   # RWrist
        coco_kps[5] = user_kps[1]   # LShoulder
        coco_kps[6] = user_kps[3]   # LElbow
        coco_kps[7] = user_kps[5]   # LWrist
        coco_kps[8]  = user_kps[8] + np.array([-0.03, 0.02])  # RHip
        coco_kps[9]  = user_kps[10]                           # RKnee
        coco_kps[10] = user_kps[10] + np.array([0.01, 0.12])  # RAnkle
        coco_kps[11] = user_kps[8] + np.array([0.03, 0.02])   # LHip
        coco_kps[12] = user_kps[9]                            # LKnee
        coco_kps[13] = user_kps[9] + np.array([-0.01, 0.12])  # LAnkle

        for idx, (start, end) in enumerate(COCO_PAIRS):
            if np.any(coco_kps[start] > 0) and np.any(coco_kps[end] > 0):
                x1, y1 = int(coco_kps[start][0] * w), int(coco_kps[start][1] * h)
                x2, y2 = int(coco_kps[end][0] * w), int(coco_kps[end][1] * h)
                cv2.line(img, (x1, y1), (x2, y2), LINE_COLORS[idx], 4)

        for idx in range(18):
            if np.any(coco_kps[idx] > 0):
                x, y = int(coco_kps[idx][0] * w), int(coco_kps[idx][1] * h)
                cv2.circle(img, (x, y), 5, JOINT_COLORS[idx], -1)
                
    return img

# ===================== 核心调整：显式分层实例掩码生成 =====================
def generate_mask(scene):
    """结合头身几何与多骨骼大半径加粗扩张，全部统一为 front_back 的显式、解耦图层渲染风格"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]
    
    if scene == "side_by_side":
        c1_x, c2_x = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["side_by_side_p1"]
        size_p2 = ELLIPSE_SIZES["side_by_side_p2"]
        
        # 1. 显式渲染 Person 2 (映射至 255) 并注入厚骨骼路径
        cv2.ellipse(mask, (c2_x, int(0.70 * IMAGE_SIZE)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(0.48 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=35)
            
        # 2. 强制覆盖叠加 Person 1 (映射至 128)
        cv2.ellipse(mask, (c1_x, int(0.70 * IMAGE_SIZE)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(0.48 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

    elif scene == "handshake":
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        
        # 1. 显式渲染 Person 2 (映射至 255) 并注入交互手部的安全 Mask 厚度
        cv2.ellipse(mask, (c2_x, int(0.70 * IMAGE_SIZE)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(0.48 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=35)
            
        # 2. 覆盖叠加 Person 1 (映射至 128)
        cv2.ellipse(mask, (c1_x, int(0.70 * IMAGE_SIZE)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(0.48 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)
            
    elif scene == "front_back":
        c1_x, c2_x = int(0.44 * IMAGE_SIZE), int(0.58 * IMAGE_SIZE)
        v_back = int(0.61 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        b_size = ELLIPSE_SIZES["front_back_back"]
        f_size = ELLIPSE_SIZES["front_back_front"]
        
        # 核心修复：原代码写死了 IMAGE_SIZE//2 造成错位。现根据 keypoints 提取精准的 c2_x 和 c1_x
        # 1. 渲染远景背景角色 (Person2 -> 255)
        cv2.ellipse(mask, (c2_x, int(v_back + 30)), (int(b_size[0]*0.8), int(b_size[1]*0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_back - 30)), int(b_size[0]*0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)
            
        # 2. 强力覆盖近景前景主体角色 (Person1 -> 128)
        cv2.ellipse(mask, (c1_x, int(v_front + 30)), (int(f_size[0]*0.8), int(f_size[1]*0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_front - 40)), int(f_size[0]*0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)
            
    return mask

# ===================== 核心调整：全肢体注入型深度图生成 =====================
def generate_depth(scene, bg):
    """全部对齐为 front_back 的显式多骨骼大半径注入逻辑，解决四肢在深度图中丢失的顽疾"""
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(IMAGE_SIZE):
        depth[i, :] = int(50 + (i / IMAGE_SIZE) * 50)
        
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
        
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]

    if scene == "side_by_side":
        c1_x, c2_x = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["side_by_side_p1"]
        size_p2 = ELLIPSE_SIZES["side_by_side_p2"]
        
        # 1. 渲染 Person 2 并彻底融入扩张的四肢骨骼 (并排场景深度值统一定为 180)
        cv2.ellipse(depth, (c2_x, int(0.70 * IMAGE_SIZE)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 180, -1)
        cv2.circle(depth, (c2_x, int(0.48 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 180, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 180, thickness=35)
            
        # 2. 渲染 Person 1 并彻底融入扩张的四肢骨骼
        cv2.ellipse(depth, (c1_x, int(0.70 * IMAGE_SIZE)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 180, -1)
        cv2.circle(depth, (c1_x, int(0.48 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 180, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 180, thickness=35)

    elif scene == "handshake":
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        
        # 1. 渲染 Person 2 并强制注入交互骨骼连线，消除深度死角
        cv2.ellipse(depth, (c2_x, int(0.70 * IMAGE_SIZE)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 180, -1)
        cv2.circle(depth, (c2_x, int(0.48 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 180, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 180, thickness=35)
            
        # 2. 渲染 Person 1 同样融合连线
        cv2.ellipse(depth, (c1_x, int(0.70 * IMAGE_SIZE)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 180, -1)
        cv2.circle(depth, (c1_x, int(0.48 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 180, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 180, thickness=35)

    elif scene == "front_back":
        b_size = ELLIPSE_SIZES["front_back_back"]
        f_size = ELLIPSE_SIZES["front_back_front"]
        
        # 1. 渲染远景角色 (Person2 -> 深度浅/值较小: 130)，位置保持 X = 0.58
        c2_x = int(0.58 * IMAGE_SIZE)
        v_back = int(0.61 * IMAGE_SIZE)
        cv2.ellipse(depth, (c2_x, int(v_back + 30)), (int(b_size[0] * 0.8), int(b_size[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(v_back - 30)), int(b_size[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)
        
        # 2. 渲染近景角色 (Person1 -> 深度深/值较大: 210)，位置保持 X = 0.44
        c1_x = int(0.44 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        cv2.ellipse(depth, (c1_x, int(v_front + 30)), (int(f_size[0] * 0.8), int(f_size[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(v_front - 40)), int(f_size[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)
            
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

# ===================== 生成 Prompt 配置 =====================
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

# ===================== 主入口 =====================
def main():
    print(" 开始生成具备高适配、防混淆特征的标准结构化测试集...")
    for folder in ["poses", "depths", "masks"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

    all_configs = []
    character_pairs = list(itertools.combinations(CHARACTERS, 2))

    for scene, bg in itertools.product(SCENES, BACKGROUNDS):
        keypoints_list = get_scene_keypoints(scene)
        pose = draw_pose_keypoints(keypoints_list, IMAGE_SIZE)
        depth = generate_depth(scene, bg)
        mask = generate_mask(scene)

        for char1, char2 in character_pairs:
            sample_id = f"{scene}_{bg}_{char1['id']}_vs_{char2['id']}"
            config = generate_prompt_config(scene, bg, char1, char2)
            config["sample_id"] = sample_id
            
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "poses", f"{sample_id}.png"), pose)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "depths", f"{sample_id}.png"), depth)
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "masks", f"{sample_id}.png"), mask)
            all_configs.append(config)
            print(f"  [+] 已输出标准对齐控制节点: {sample_id}")

    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
        
    print(f"\n 标准化合成测试集已构建成功！样本总规模: {len(all_configs)} 组。")

if __name__ == "__main__":
    main()