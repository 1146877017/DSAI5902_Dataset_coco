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

ELLIPSE_SIZES = {
    "side_by_side": (75, 180),
    "handshake": (75, 180),
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
    """统一管理场景关键点，确保真正完美的正面视角（人体结构左侧在图像右，右侧在图像左）"""
    if scene == "side_by_side":
        # Person 1 (左侧角色，中心 X=0.30)
        kps_p1 = np.array([
            [0.30, 0.50],  # 0: Nose
            [0.34, 0.58],  # 1: LShoulder (正面视角下，左肩偏右侧 X=0.34)
            [0.26, 0.58],  # 2: RShoulder (右肩偏左侧 X=0.26)
            [0.36, 0.68],  # 3: LElbow
            [0.24, 0.68],  # 4: RElbow
            [0.38, 0.78],  # 5: LWrist
            [0.22, 0.78],  # 6: RWrist
            [0.30, 0.54],  # 7: Neck
            [0.30, 0.72],  # 8: Pelvis
            [0.33, 0.84],  # 9: LKnee
            [0.27, 0.84]   # 10: RKnee
        ])
        # Person 2 (右侧角色，中心 X=0.70)
        kps_p2 = np.array([
            [0.70, 0.50],  # 0: Nose
            [0.74, 0.58],  # 1: LShoulder
            [0.66, 0.58],  # 2: RShoulder
            [0.76, 0.68],  # 3: LElbow
            [0.64, 0.68],  # 4: RElbow
            [0.78, 0.78],  # 5: LWrist
            [0.62, 0.78],  # 6: RWrist
            [0.70, 0.54],  # 7: Neck
            [0.70, 0.72],  # 8: Pelvis
            [0.73, 0.84],  # 9: LKnee
            [0.67, 0.84]   # 10: RKnee
        ])
    elif scene == "handshake":
        # 两人相向站立并伸手交叉握手（完美正面透视）
        # 精准重合点依然维持在图像正中 [0.50, 0.62]
        kps_p1 = np.array([
            [0.35, 0.48],  # 0: Nose
            [0.40, 0.55],  # 1: LShoulder
            [0.30, 0.55],  # 2: RShoulder
            [0.43, 0.65],  # 3: LElbow
            [0.38, 0.60],  # 4: RElbow
            [0.45, 0.75],  # 5: LWrist
            [0.50, 0.62],  # 6: RWrist (P1的右手臂向前伸到中间握手)
            [0.35, 0.51],  # 7: Neck
            [0.35, 0.72],  # 8: Pelvis
            [0.38, 0.85],  # 9: LKnee
            [0.32, 0.85]   # 10: RKnee
        ])
        kps_p2 = np.array([
            [0.65, 0.48],  # 0: Nose
            [0.70, 0.55],  # 1: LShoulder
            [0.60, 0.55],  # 2: RShoulder
            [0.58, 0.60],  # 3: LElbow (P2的左手臂向前伸到中间握手)
            [0.57, 0.65],  # 4: RElbow
            [0.50, 0.62],  # 5: LWrist
            [0.55, 0.75],  # 6: RWrist
            [0.65, 0.51],  # 7: Neck
            [0.65, 0.72],  # 8: Pelvis
            [0.68, 0.85],  # 9: LKnee
            [0.62, 0.85]   # 10: RKnee
        ])
    else:  # front_back (前后景深错开完美透视)
        # Person 1 (近景主体：整体放大，面向镜头正面)
        kps_p1 = np.array([
            [0.44, 0.42],  # 0: Nose
            [0.50, 0.52],  # 1: LShoulder
            [0.38, 0.52],  # 2: RShoulder
            [0.54, 0.65],  # 3: LElbow
            [0.34, 0.65],  # 4: RElbow
            [0.56, 0.78],  # 5: LWrist
            [0.32, 0.78],  # 6: RWrist
            [0.44, 0.48],  # 7: Neck
            [0.44, 0.72],  # 8: Pelvis
            [0.48, 0.88],  # 9: LKnee
            [0.40, 0.88]   # 10: RKnee
        ])
        # Person 2 (远景背景：整体缩小，面向镜头正面)
        kps_p2 = np.array([
            [0.58, 0.35],  # 0: Nose
            [0.62, 0.42],  # 1: LShoulder
            [0.54, 0.42],  # 2: RShoulder
            [0.65, 0.52],  # 3: LElbow
            [0.51, 0.52],  # 4: RElbow
            [0.67, 0.62],  # 5: LWrist
            [0.49, 0.62],  # 6: RWrist
            [0.58, 0.39],  # 7: Neck
            [0.58, 0.58],  # 8: Pelvis
            [0.61, 0.72],  # 9: LKnee
            [0.55, 0.72]   # 10: RKnee
        ])
    return [kps_p1, kps_p2]

# =====================兼容正面级官方标准 OpenPose 渲染器 =====================
def draw_pose_keypoints(keypoints_list, img_size=512):
    """动态补充双眼、双耳的标准面部连线，从底层逻辑封死背影漏洞，强力引导模型生成正面"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    
    # 官方标准 COCO 肢体与面部完整有效连线
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
        
        # 调整下半身正面拓扑映射，使胯部/双脚骨骼不会发生交叉错位扭曲
        coco_kps[8]  = user_kps[8] + np.array([-0.03, 0.02])  # RHip (偏左)
        coco_kps[9]  = user_kps[10]                               # RKnee (偏左)
        coco_kps[10] = user_kps[10] + np.array([-0.01, 0.12]) # RAnkle (偏左)
        coco_kps[11] = user_kps[8] + np.array([0.03, 0.02])    # LHip (偏右)
        coco_kps[12] = user_kps[9]                             # LKnee (偏右)
        coco_kps[13] = user_kps[9] + np.array([0.01, 0.12])   # LAnkle (偏右)

        # 根据头颈比例，自动计算并填充正面标准的五官点 (眼睛和耳朵)
        # 正面视角：右侧器官在画面的左边 (-dx)，左侧器官在画面的右边 (+dx)
        head_scale = np.linalg.norm(coco_kps[0] - coco_kps[1]) if np.linalg.norm(coco_kps[0] - coco_kps[1]) > 0 else 0.05
        dx_eye, dy_eye = head_scale * 0.35, head_scale * 0.15
        dx_ear, dy_ear = head_scale * 0.70, head_scale * 0.10
        
        coco_kps[14] = coco_kps[0] + np.array([-dx_eye, -dy_eye]) # 14: REye (画质偏左)
        coco_kps[15] = coco_kps[0] + np.array([dx_eye, -dy_eye])  # 15: LEye (画质偏右)
        coco_kps[16] = coco_kps[0] + np.array([-dx_ear, dy_ear])  # 16: REar
        coco_kps[17] = coco_kps[0] + np.array([dx_ear, dy_ear])   # 17: LEar

        # 渲染肢体连接线
        for idx, (start, end) in enumerate(COCO_PAIRS):
            if np.any(coco_kps[start] > 0) and np.any(coco_kps[end] > 0):
                x1, y1 = int(coco_kps[start][0] * w), int(coco_kps[start][1] * h)
                x2, y2 = int(coco_kps[end][0] * w), int(coco_kps[end][1] * h)
                cv2.line(img, (x1, y1), (x2, y2), LINE_COLORS[idx], 4)

        # 渲染关节点圆圈
        for idx in range(18):
            if np.any(coco_kps[idx] > 0):
                x, y = int(coco_kps[idx][0] * w), int(coco_kps[idx][1] * h)
                cv2.circle(img, (x, y), 5, JOINT_COLORS[idx], -1)
                
    return img

# ===================== 四肢骨骼大半径扩张型实例掩码生成 =====================
def generate_mask(scene):
    """结合头身几何与多骨骼动态扩张，确保正面肢体掩码与骨骼完美对齐，消除大头畸变"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]
    
    # 为了保证前后景景深遮挡正确，front_back 场景先画远景(Person 2)，再画近景(Person 1)
    loop_order = [1, 0] if scene == "front_back" else [0, 1]
    
    for i in loop_order:
        kps = keypoints_list[i]
        lbl = 128 if i == 0 else 255
        
        # 提取骨骼点并转换到像素坐标
        nose = kps[0] * IMAGE_SIZE
        neck = kps[7] * IMAGE_SIZE
        pelvis = kps[8] * IMAGE_SIZE
        l_shoulder = kps[1] * IMAGE_SIZE
        r_shoulder = kps[2] * IMAGE_SIZE
        
        # 1. 动态计算科学的头部大小与中心（依据鼻子到脖子的距离）
        head_height = np.linalg.norm(nose - neck)
        head_center = nose + (nose - neck) * 0.1  # 中心略微上移涵盖颅顶
        head_radius = int(head_height * 0.85)     # 动漫/写实人体黄金比例半径
        
        # 2. 动态计算身体椭圆大小与中心（依据肩宽与躯干高度）
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        torso_height = np.linalg.norm(neck - pelvis)
        body_center = (neck + pelvis) / 2
        body_axes = (int(shoulder_width * 0.65), int(torso_height * 0.55))
        
        # 3. 渲染身体躯干与头部
        cv2.ellipse(mask, (int(body_center[0]), int(body_center[1])), body_axes, 0, 0, 360, lbl, -1)
        cv2.circle(mask, (int(head_center[0]), int(head_center[1])), head_radius, lbl, -1)
        
        # 4. 肢体骨骼线粗化扩张
        thickness = 35 if (scene != "front_back" or i == 0) else 25
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(kps[start][0] * IMAGE_SIZE), int(kps[start][1] * IMAGE_SIZE)
            x2, y2 = int(kps[end][0] * IMAGE_SIZE), int(kps[end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), lbl, thickness=thickness)
            
    return mask


# ===================== 正面深度图基础生成逻辑 =====================
def generate_depth(scene, bg):
    """基于 OpenPose 关键点动态构建三维深度躯干"""
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    # 1. 渲染背景线条与渐变
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
        
    # 2. 动态计算人物深度几何体
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]
    
    # 严格遵循先远后近的渲染顺序
    loop_order = [1, 0] if scene == "front_back" else [0, 1]
    
    for i in loop_order:
        kps = keypoints_list[i]
        val = 180 if scene in ["side_by_side", "handshake"] else (210 if i == 0 else 130)
            
        # 提取骨骼点像素坐标
        nose = kps[0] * IMAGE_SIZE
        neck = kps[7] * IMAGE_SIZE
        pelvis = kps[8] * IMAGE_SIZE
        l_shoulder = kps[1] * IMAGE_SIZE
        r_shoulder = kps[2] * IMAGE_SIZE
        
        # 动态计算头部（使头半径缩减到约 18-20 像素）
        head_height = np.linalg.norm(nose - neck)
        head_center = nose + (nose - neck) * 0.1
        head_radius = int(head_height * 0.85)
        
        # 动态计算躯干椭圆
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
        torso_height = np.linalg.norm(neck - pelvis)
        body_center = (neck + pelvis) / 2
        body_axes = (int(shoulder_width * 0.65), int(torso_height * 0.55))
        
        # 渲染深度轮廓
        cv2.ellipse(depth, (int(body_center[0]), int(body_center[1])), body_axes, 0, 0, 360, val, -1)
        cv2.circle(depth, (int(head_center[0]), int(head_center[1])), head_radius, val, -1)
        
        # 渲染四肢渐变/衔接骨骼深度线
        thickness = 30 if (scene != "front_back" or i == 0) else 20
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(kps[start][0] * IMAGE_SIZE), int(kps[start][1] * IMAGE_SIZE)
            x2, y2 = int(kps[end][0] * IMAGE_SIZE), int(kps[end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), val, thickness=thickness)
            
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)



def generate_prompt_config(scene, bg, char1, char2):
    p1_tags = char1["triggers"]
    p2_tags = char2["triggers"]
    prompt = f"person1: a close-up front photo of {p1_tags}. person2: a close-up front photo of {p2_tags}. {scene.replace('_', ' ')} scene facing camera in a {bg.replace('_', ' ')}, high quality, 8k, realistic"
    neg_prompt = "blurry, low quality, distorted, missing people, extra limbs, monochrome, side view, back view, from behind"
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
    print(" Starting to generate a standard structured test dataset with features of [high adaptability, anti-confusion, absolute frontal orientation]...")
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
            print(f"  [+] Fully-aligned frontal control nodes outputted: {sample_id}")

    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
        
    print(f"\n Standardized synthetic frontal test dataset built successfully! Total sample size: {len(all_configs)} groups.")

if __name__ == "__main__":
    main()