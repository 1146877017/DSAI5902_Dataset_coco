import os
import cv2
import numpy as np
import json
import itertools

# ===================== 全局配置 =====================
print("[INIT] Loading global configuration...")
OUTPUT_ROOT = "synthetic_test_dataset"
# 扩展 side_by_side 为 3 帧连续动作序列，其余场景保持不变
SCENES = [
    "side_by_side_stand",      # 序列帧1：两人自然站立（原基准姿态）
    "side_by_side_raise_hand", # 序列帧2：左侧人物抬手
    "side_by_side_point",      # 序列帧3：左侧人物指向右侧人物
    "handshake",
    "front_back"
]
BACKGROUNDS = ["outdoor_park", "indoor_room", "futuristic_street"]
IMAGE_SIZE = 512
print(f"[INIT] SCENES count: {len(SCENES)}, BACKGROUNDS count: {len(BACKGROUNDS)}, IMAGE_SIZE: {IMAGE_SIZE}")

# 所有场景统一近大远小：近景尺寸大，远景尺寸小，匹配深度层级
# side_by_side 三帧序列共用同一套人物尺寸（躯干大小完全一致，仅手臂动作变化）
ELLIPSE_SIZES = {
    "side_by_side_p1": (80, 190),    # 近景 左侧人物
    "side_by_side_p2": (70, 170),    # 远景 右侧人物
    "handshake_p1": (80, 190),
    "handshake_p2": (70, 170),
    "front_back_back": (85, 130),
    "front_back_front": (110, 170)
}
print(f"[INIT] ELLIPSE_SIZES loaded: {list(ELLIPSE_SIZES.keys())}")

# ===================== 4 个 LoRA 角色模型信息 =====================
CHARACTERS = [
    {
        "id": "Sera",
        "lora_name": "LoRA_Sera",
        "triggers": "SeraDef, brown hair, red dress, egyptian, hair tubes, bare shoulders, ankh, jewelry, collarbone"
    },
    {
        "id": "TogaHimiko",
        "lora_name": "TogaHimiko-01",
        "triggers": "Himiko Toga, blonde hair, school uniform, beige cardigan, Toga, Himiko, black stockings"
    },
    {
        "id": "MouriRan",
        "lora_name": "Mouri",
        "triggers": "mouriranai, mouri ran, long sleeves, blue eyes, school uniform, blue jacket, green necktie, pleated skirt, meitantei conan"
    },
    {
        "id": "Byakuya",
        "lora_name": "Byakuya",
        "triggers": "Byakuyadef, 1girl, dark hair, long hair, uniform, green bow tie" 
    }
]
print(f"[INIT] CHARACTERS loaded: {[c['id'] for c in CHARACTERS]} (total {len(CHARACTERS)})")

# ===================== 场景关键点映射器（含连续动作序列） =====================
def get_scene_keypoints(scene):
    print(f"  [FUNC] get_scene_keypoints called, scene={scene}")
    # ========== side_by_side 连续动作序列 ==========
    # 三帧共用 p2 关键点（右侧人物完全不动），仅 p1 右臂动作变化
    # 人物躯干、头部、腿部位置完全一致，保证空间站位稳定
    p2_side_common = np.array([
        [0.70, 0.42], [0.64, 0.52], [0.76, 0.52],
        [0.60, 0.65], [0.80, 0.65], [0.58, 0.78], [0.82, 0.78],
        [0.70, 0.49], [0.70, 0.72], [0.64, 0.88], [0.76, 0.88]
    ])

    if scene == "side_by_side_stand":
        # 帧1：自然站立（原基准姿态）
        print(f"  [FUNC] matched scene: side_by_side_stand (frame 1 - neutral)")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.40, 0.65], [0.18, 0.78], [0.42, 0.78],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        print(f"  [FUNC] returning keypoints: p1 shape={p1.shape}, p2 shape={p2_side_common.shape}")
        return [p1, p2_side_common]

    elif scene == "side_by_side_raise_hand":
        # 帧2：左侧人物抬起右手（手肘弯曲，手举至头部右侧）
        # 仅修改右肘(4)、右腕(6)坐标，其余点完全不变
        print(f"  [FUNC] matched scene: side_by_side_raise_hand (frame 2 - raise right hand)")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.38, 0.45], [0.18, 0.78], [0.35, 0.38],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        print(f"  [FUNC] returning keypoints: p1 shape={p1.shape}, p2 shape={p2_side_common.shape}")
        return [p1, p2_side_common]

    elif scene == "side_by_side_point":
        # 帧3：左侧人物伸直右臂，指向右侧人物（手臂水平前伸）
        # 仅修改右肘(4)、右腕(6)坐标，其余点完全不变
        print(f"  [FUNC] matched scene: side_by_side_point (frame 3 - point right)")
        p1 = np.array([
            [0.30, 0.42], [0.24, 0.52], [0.36, 0.52],
            [0.20, 0.65], [0.46, 0.54], [0.18, 0.78], [0.56, 0.52],
            [0.30, 0.49], [0.30, 0.72], [0.24, 0.88], [0.36, 0.88]
        ])
        print(f"  [FUNC] returning keypoints: p1 shape={p1.shape}, p2 shape={p2_side_common.shape}")
        return [p1, p2_side_common]

    elif scene == "handshake":
        print(f"  [FUNC] matched scene: handshake")
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
        print(f"  [FUNC] returning keypoints: p1 shape={kps_p1.shape}, p2 shape={kps_p2.shape}")
        return [kps_p1, kps_p2]

    else:  # front_back
        print(f"  [FUNC] matched scene: front_back (fallback)")
        kps_p1 = np.array([
            [0.44, 0.42], [0.38, 0.52], [0.50, 0.52], [0.34, 0.65], [0.54, 0.65],
            [0.32, 0.78], [0.56, 0.78], [0.44, 0.49], [0.44, 0.72], [0.40, 0.88], [0.48, 0.88]
        ])
        kps_p2 = np.array([
            [0.58, 0.35], [0.54, 0.42], [0.62, 0.42], [0.51, 0.52], [0.65, 0.52],
            [0.49, 0.62], [0.67, 0.62], [0.58, 0.40], [0.58, 0.58], [0.55, 0.72], [0.61, 0.72]
        ])
        print(f"  [FUNC] returning keypoints: p1 shape={kps_p1.shape}, p2 shape={kps_p2.shape}")
        return [kps_p1, kps_p2]

# ===================== 标准多色 OpenPose 骨骼图生成 =====================
def draw_pose_keypoints(keypoints_list, img_size=512):
    print(f"  [FUNC] draw_pose_keypoints called, num_persons={len(keypoints_list)}, img_size={img_size}")
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

    for person_idx, user_kps in enumerate(keypoints_list):
        print(f"  [FUNC] drawing person {person_idx}, keypoints shape={user_kps.shape}")
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

    print(f"  [FUNC] draw_pose_keypoints done, output shape={img.shape}")
    return img

# ===================== 掩码生成 =====================
def generate_mask(scene):
    print(f"  [FUNC] generate_mask called, scene={scene}")
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]

    # side_by_side 连续序列统一处理（三帧躯干位置/大小完全一致）
    if scene.startswith("side_by_side"):
        print(f"  [FUNC] mask branch: side_by_side series")
        c1_x, c2_x = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["side_by_side_p1"]  # 近景
        size_p2 = ELLIPSE_SIZES["side_by_side_p2"]  # 远景
        v_body = int(0.61 * IMAGE_SIZE)
        v_head = int(0.42 * IMAGE_SIZE)
        print(f"  [FUNC] mask sizes: p1={size_p1} (near, 128), p2={size_p2} (far, 255)")

        # 底层：远景 Person2 (255 右侧)
        print(f"  [FUNC] drawing far person (p2, value=255) as bottom layer")
        cv2.ellipse(mask, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_head - 4)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

        # 上层：近景 Person1 (128 左侧)，重叠区保留近景
        print(f"  [FUNC] drawing near person (p1, value=128) as top layer")
        cv2.ellipse(mask, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_head - 4)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

    elif scene == "handshake":
        print(f"  [FUNC] mask branch: handshake")
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        v_body = int(0.61 * IMAGE_SIZE)
        v_head = int(0.42 * IMAGE_SIZE)
        print(f"  [FUNC] mask sizes: p1={size_p1}, p2={size_p2}")

        cv2.ellipse(mask, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_head - 4)), int(size_p2[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

        cv2.ellipse(mask, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_head - 4)), int(size_p1[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

    elif scene == "front_back":
        print(f"  [FUNC] mask branch: front_back")
        c1_x, c2_x = int(0.44 * IMAGE_SIZE), int(0.58 * IMAGE_SIZE)
        v_back = int(0.61 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        b_size = ELLIPSE_SIZES["front_back_back"]
        f_size = ELLIPSE_SIZES["front_back_front"]
        print(f"  [FUNC] mask sizes: back={b_size} (255), front={f_size} (128)")

        cv2.ellipse(mask, (c2_x, int(v_back + 30)), (int(b_size[0] * 0.8), int(b_size[1] * 0.7)), 0, 0, 360, 255, -1)
        cv2.circle(mask, (c2_x, int(v_back - 30)), int(b_size[0] * 0.5), 255, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=25)

        cv2.ellipse(mask, (c1_x, int(v_front + 30)), (int(f_size[0] * 0.8), int(f_size[1] * 0.7)), 0, 0, 360, 128, -1)
        cv2.circle(mask, (c1_x, int(v_front - 40)), int(f_size[0] * 0.5), 128, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(mask, (x1, y1), (x2, y2), 128, thickness=35)

    print(f"  [FUNC] generate_mask done, mask shape={mask.shape}, unique values={np.unique(mask).tolist()}")
    return mask

# ===================== 深度图生成 =====================
def generate_depth(scene, bg):
    print(f"  [FUNC] generate_depth called, scene={scene}, bg={bg}")
    depth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(IMAGE_SIZE):
        depth[i, :] = int(50 + (i / IMAGE_SIZE) * 50)
    print(f"  [FUNC] base gradient depth range: [{depth.min()}, {depth.max()}]")

    if bg == "indoor_room":
        print(f"  [FUNC] adding indoor_room background depth details")
        cv2.line(depth, (0, 0), (120, 150), 80, 2)
        cv2.line(depth, (IMAGE_SIZE, 0), (IMAGE_SIZE-120, 150), 80, 2)
        cv2.line(depth, (120, 150), (IMAGE_SIZE-120, 150), 70, 2)
    elif bg == "outdoor_park":
        print(f"  [FUNC] adding outdoor_park background depth details")
        pts = np.array([[0, 250], [100, 180], [250, 230], [400, 150], [IMAGE_SIZE, 220], [IMAGE_SIZE, 300], [0, 300]], np.int32)
        cv2.fillPoly(depth, [pts], 75)
    elif bg == "futuristic_street":
        print(f"  [FUNC] adding futuristic_street background depth details")
        cv2.rectangle(depth, (0, 0), (80, 400), 65, -1)
        cv2.rectangle(depth, (IMAGE_SIZE-80, 0), (IMAGE_SIZE, 400), 65, -1)

    keypoints_list = get_scene_keypoints(scene)
    USER_CONNECTIONS = [(0, 7), (7, 1), (7, 2), (1, 3), (2, 4), (3, 5), (4, 6), (7, 8), (8, 9), (8, 10)]

    # side_by_side 连续序列统一处理
    if scene.startswith("side_by_side"):
        print(f"  [FUNC] depth branch: side_by_side series")
        c1_x, c2_x = int(0.30 * IMAGE_SIZE), int(0.70 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["side_by_side_p1"]
        size_p2 = ELLIPSE_SIZES["side_by_side_p2"]
        v_body = int(0.61 * IMAGE_SIZE)

        # 底层：远景 Person2 (深度130)
        print(f"  [FUNC] drawing far person (p2, depth=130)")
        cv2.ellipse(depth, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(0.42 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)

        # 上层：近景 Person1 (深度210)
        print(f"  [FUNC] drawing near person (p1, depth=210)")
        cv2.ellipse(depth, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(0.42 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)

    elif scene == "handshake":
        print(f"  [FUNC] depth branch: handshake")
        c1_x, c2_x = int(0.35 * IMAGE_SIZE), int(0.65 * IMAGE_SIZE)
        size_p1 = ELLIPSE_SIZES["handshake_p1"]
        size_p2 = ELLIPSE_SIZES["handshake_p2"]
        v_body = int(0.61 * IMAGE_SIZE)

        cv2.ellipse(depth, (c2_x, int(v_body + 30)), (int(size_p2[0] * 0.8), int(size_p2[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(0.42 * IMAGE_SIZE)), int(size_p2[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)

        cv2.ellipse(depth, (c1_x, int(v_body + 30)), (int(size_p1[0] * 0.8), int(size_p1[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(0.42 * IMAGE_SIZE)), int(size_p1[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)

    elif scene == "front_back":
        print(f"  [FUNC] depth branch: front_back")
        b_size = ELLIPSE_SIZES["front_back_back"]
        f_size = ELLIPSE_SIZES["front_back_front"]

        c2_x = int(0.58 * IMAGE_SIZE)
        v_back = int(0.61 * IMAGE_SIZE)
        cv2.ellipse(depth, (c2_x, int(v_back + 30)), (int(b_size[0] * 0.8), int(b_size[1] * 0.7)), 0, 0, 360, 130, -1)
        cv2.circle(depth, (c2_x, int(v_back - 30)), int(b_size[0] * 0.5), 130, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[1][start][0] * IMAGE_SIZE), int(keypoints_list[1][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[1][end][0] * IMAGE_SIZE), int(keypoints_list[1][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 130, thickness=25)

        c1_x = int(0.44 * IMAGE_SIZE)
        v_front = int(0.71 * IMAGE_SIZE)
        cv2.ellipse(depth, (c1_x, int(v_front + 30)), (int(f_size[0] * 0.8), int(f_size[1] * 0.7)), 0, 0, 360, 210, -1)
        cv2.circle(depth, (c1_x, int(v_front - 40)), int(f_size[0] * 0.5), 210, -1)
        for (start, end) in USER_CONNECTIONS:
            x1, y1 = int(keypoints_list[0][start][0] * IMAGE_SIZE), int(keypoints_list[0][start][1] * IMAGE_SIZE)
            x2, y2 = int(keypoints_list[0][end][0] * IMAGE_SIZE), int(keypoints_list[0][end][1] * IMAGE_SIZE)
            cv2.line(depth, (x1, y1), (x2, y2), 210, thickness=35)

    print(f"  [FUNC] applying Gaussian blur (kernel=15) and converting to BGR")
    depth = cv2.GaussianBlur(depth, (15, 15), 0)
    result = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    print(f"  [FUNC] generate_depth done, output shape={result.shape}, depth range: [{result.min()}, {result.max()}]")
    return result

# ===================== 生成 Prompt 配置 =====================
def generate_prompt_config(scene, bg, char1, char2):
    print(f"  [FUNC] generate_prompt_config called, scene={scene}, bg={bg}, char1={char1['id']}, char2={char2['id']}")
    p1_tags = char1["triggers"]
    p2_tags = char2["triggers"]
    
    # 对 side_by_side 序列统一场景描述，仅动作差异由 ControlNet 控制
    scene_display = scene.replace('_', ' ')
    prompt = f"person1: a photo of {p1_tags}. person2: a photo of {p2_tags}. {scene_display} scene in a {bg.replace('_', ' ')}, high quality, 8k, realistic"
    neg_prompt = "blurry, low quality, distorted, missing people, extra limbs, monochrome"
    
    config = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "scene": scene,
        "background": bg,
        "characters": [char1["id"], char2["id"]],
        "sequence_group": "side_by_side" if scene.startswith("side_by_side") else scene,
        "lora_details": [
            {"character_id": char1["id"], "lora_name": char1["lora_name"], "weight": 0.8},
            {"character_id": char2["id"], "lora_name": char2["lora_name"], "weight": 0.8}
        ]
    }
    print(f"  [FUNC] generate_prompt_config done, sequence_group={config['sequence_group']}")
    return config

# ===================== 主入口 =====================
def main():
    print("=" * 60)
    print("[MAIN] Begin generating a standardized structured test set with narrative sequence support...")
    print("=" * 60)

    print("[MAIN] Creating output directories...")
    for folder in ["poses", "depths", "masks"]:
        path = os.path.join(OUTPUT_ROOT, folder)
        os.makedirs(path, exist_ok=True)
        print(f"[MAIN] ensured directory: {path}")

    all_configs = []
    character_pairs = list(itertools.combinations(CHARACTERS, 2))
    print(f"[MAIN] character pairs generated: {len(character_pairs)} pairs")
    for i, (c1, c2) in enumerate(character_pairs):
        print(f"  pair {i+1}: {c1['id']} vs {c2['id']}")

    total_scene_bg = len(SCENES) * len(BACKGROUNDS)
    total_expected = total_scene_bg * len(character_pairs)
    print(f"[MAIN] Expected total samples: {total_expected} (scenes*backgrounds={total_scene_bg} * pairs={len(character_pairs)})")

    scene_bg_idx = 0
    for scene, bg in itertools.product(SCENES, BACKGROUNDS):
        scene_bg_idx += 1
        print(f"\n[MAIN] === Processing [{scene_bg_idx}/{total_scene_bg}]: scene={scene}, bg={bg} ===")

        print("[MAIN] Generating keypoints...")
        keypoints_list = get_scene_keypoints(scene)

        print("[MAIN] Drawing pose image...")
        pose = draw_pose_keypoints(keypoints_list, IMAGE_SIZE)

        print("[MAIN] Generating depth image...")
        depth = generate_depth(scene, bg)

        print("[MAIN] Generating mask image...")
        mask = generate_mask(scene)

        print(f"[MAIN] Iterating over {len(character_pairs)} character pairs...")
        for pair_idx, (char1, char2) in enumerate(character_pairs):
            sample_id = f"{scene}_{bg}_{char1['id']}_vs_{char2['id']}"
            print(f"  [MAIN] Pair {pair_idx+1}/{len(character_pairs)}: {sample_id}")

            print(f"    generating prompt config...")
            config = generate_prompt_config(scene, bg, char1, char2)
            config["sample_id"] = sample_id

            pose_path = os.path.join(OUTPUT_ROOT, "poses", f"{sample_id}.png")
            depth_path = os.path.join(OUTPUT_ROOT, "depths", f"{sample_id}.png")
            mask_path = os.path.join(OUTPUT_ROOT, "masks", f"{sample_id}.png")

            print(f"    saving pose to: {pose_path}")
            cv2.imwrite(pose_path, pose)
            print(f"    saving depth to: {depth_path}")
            cv2.imwrite(depth_path, depth)
            print(f"    saving mask to: {mask_path}")
            cv2.imwrite(mask_path, mask)

            all_configs.append(config)
            print(f"    [+] Generated sample: {sample_id} (total so far: {len(all_configs)})")

    print("\n" + "=" * 60)
    print("[MAIN] All samples generated. Saving JSON config...")
    config_output_path = os.path.join(OUTPUT_ROOT, "synthetic_configs.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=4, ensure_ascii=False)
    print(f"[MAIN] Config saved to: {config_output_path}")

    # 统计序列信息
    seq_count = sum(1 for s in SCENES if s.startswith("side_by_side"))
    print(f"\n[MAIN] Test set construction complete!")
    print(f"[MAIN] Total samples: {len(all_configs)}")
    print(f"[MAIN] Side-by-side narrative frames: {seq_count} frames per character pair per background")
    print("=" * 60)

if __name__ == "__main__":
    main()