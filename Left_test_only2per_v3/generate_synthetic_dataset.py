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

# 将所有场景的角色尺寸全部解耦对齐，完全统一front_back规范
ELLIPSE_SIZES = {
    "side_by_side_p1": (75, 180),
    "side_by_side_p2": (75, 180),
    "handshake_p1": (75, 180),
    "handshake_p2": (75, 180),
    "front_back_back": (85, 130),
    "front_back_front": (110, 170)
}

# ===================== 配置： 4 个 LoRA 角色模型信息  =====================
CHARACTERS = [
    {
        "id": "Sera",
        "lora_name": "LoRA_Sera",
        # 核心触发词 + 极具辨识度的埃及/暗肤色/红裙特征，剔除过于细碎的饰品词以防注意力稀释
        "triggers": "SeraDef, dark-skinned female, red dress, egyptian, hair tubes, bare shoulders, ankh, jewelry"
    },
    {
        "id": "TogaHimiko",
        "lora_name": "TogaHimiko-01",
        # 核心触发词 + 标志性的金发乱发与开衫校服特征
        "triggers": "Himiko Toga, blonde hair, messy hair, school uniform, beige cardigan"
    },
    {
        "id": "MouriRan",
        "lora_name": "Mouri",
        # 核心触发词 + 标志性校服与发型，已移除 lens flare/open mouth 等干扰词
        "triggers": "mouriranai, mouri ran, brown hair, blue eyes, school uniform, blue jacket, green necktie, pleated skirt"
    },
    {
        "id": "Byakuya",
        "lora_name": "Byakuya",
        # 核心触发词 + 基础外观锚点（黑发/制服），帮助 CLIP 更好地定位角色基础结构
        "triggers": "Byakuyadef, 1girl, dark hair, short hair, uniform" 
    }
]

# ===================== 统一的场景坐标映射器（拉高头部Y，和front_back一致舒展正面全身） =====================
def get_scene_keypoints(scene):
    """统一管理场景关键点，全部对齐front_back纵向高度，人物头部抬高，杜绝低矮畸形、背对"""
    if scene == "side_by_side":
        # 纵向Y坐标抬高，和front_back头部高度统一
        kps_p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.40, 0.65], [0.18, 0.78], [0.42, 0.78],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        kps_p2 = np.array([
            [0.70, 0.42], [0.64, 0.52], [0.76, 0.52],
            [0.60, 0.65], [0.80, 0.65], [0.58, 0.78], [0.82, 0.78],
            [0.70, 0.49], [0.70, 0.72], [0.64, 0.88], [0.76, 0.88]
        ])
    elif scene == "handshake":
        kps_p1 = np.array([
            [0.35, 0.42], [0.29, 0.52], [0.41, 0.52],
            [0.25, 0.65], [0.48, 0.60], [0.22, 0.78], [0.50, 0.62],
            [0.35, 0.49], [0.35, 0.72], [0.30, 0.88], [0.40, 0.88]
        ])
        kps_p2 = np.array([
            [0.65, 0.42], [0.59, 0.52], [0.71, 0.52],
            [0.52, 0.60], [0.75, 0.65], [0.50, 0.62], [0.78, 0.78],
            [0.65, 0.49], [0.65, 0.72], [0.60, 0.88], [0.70, 0.88]
        ])
    else:  # front_back 原始最优坐标不变
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
        coco_kps[8] = user_kps[8] + np.array([-0.03, 0.02])  # RHip
        coco_kps[9] = user_kps[10]                           # RKnee
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

# ===================== 核心修改：side_by_side / handshake 交换绘制顺序，先p1(128)后p2(255) =====================
def generate_mask(scene):
    """side_by_side/handshake：先person1(128底层)，后person2(255上层)，重叠区保留右侧人物掩码"""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]

    if scene == "side_by_side":
        c1_x, c2_x = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["side_by_side_p1"]
        size_p2 = ELLIPSE_SIZES["side_by_side_p2"]
        v_body = int(0.61 * IMAGE_SIZE)
        v_head = int(0.42 * IMAGE_SIZE)

        # 【修改1】先画底层 Person1 (128 左侧)
        cv2.ellipse(mask, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_head - 4)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

        # 【修改1】后画上层 Person2 (255 右侧)，重叠区域保留255，不会被覆盖丢失
        cv2.ellipse(mask, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_head - 4)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

    elif scene == "handshake":
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        v_body = int(0.61 * IMAGE_SIZE)
        v_head = int(0.42 * IMAGE_SIZE)

        # 【修改2】先底层 Person1 (128)
        cv2.ellipse(mask, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_head - 4)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

        # 【修改2】后上层 Person2 (255)
        cv2.ellipse(mask, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_head - 4)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

    elif scene == "front_back":
        # front_back 远近分层逻辑不变，不需要调换顺序
        c1_x, c2_x = int(0.44 * IMAGE_SIZE), int(0.58 * IMAGE_SIZE)
        v_back = int(0.61 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        b_size = ELLIPSE_SIZES["front_back_back"]
        f_size = ELLIPSE_SIZES["front_back_front"]

        # 1. 渲染远景背景角色 (Person2 -> 255)
        cv2.ellipse(mask, (c2_x, int(v_back + 30)), (int(b_size[0] * 0.8), int(b_size[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_back - 30)), int(b_size[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

        # 2. 强力覆盖近景前景主体角色 (Person1 -> 128)
        cv2.ellipse(mask, (c1_x, int(v_front + 30)), (int(f_size[0] * 0.8), int(f_size[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_front - 40)), int(f_size[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

    return mask

# ===================== 深度图统一：分层数值、骨骼粗细、模糊全部对齐front_back =====================
def generate_depth(scene, bg):
    """side_by_side / handshake 深度分层、线条粗细、高斯核完全复用front_back参数"""
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
        v_body = int(0.61 * IMAGE_SIZE)

        # 深度同步调换绘制顺序：先p1(210)底层，后p2(130)上层
        cv2.ellipse(depth, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(0.42 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)

        cv2.ellipse(depth, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(0.42 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)

    elif scene == "handshake":
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        v_body = int(0.61 * IMAGE_SIZE)

        # 深度同步调换绘制顺序：先p1(210)底层，后p2(130)上层
        cv2.ellipse(depth, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(0.42 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)

        cv2.ellipse(depth, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(0.42 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)

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
    print(" Begin generating a standardized structured test set with high adaptability and anti-confusion features...")
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
            print(f"  [+] Standard alignment control node output : {sample_id}")

    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)

    print(f"\n The standardized synthetic test set has been successfully constructed! Total sample size : {len(all_configs)} ")

if __name__ == "__main__":
    main()