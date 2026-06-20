import os
import cv2
import numpy as np
from config import (
    CONDITION_DIR, IMAGE_WIDTH, IMAGE_HEIGHT,
    SCENES, BACKGROUNDS, CHARACTER_PAIRS
)

def generate_side_by_side_conditions():
    """生成并排站立场景的掩码与深度图"""
    # 左右角色二值掩码
    mask_a = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask_b = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask_a[:, :IMAGE_WIDTH//2] = 255  # 左半区
    mask_b[:, IMAGE_WIDTH//2:] = 255  # 右半区

    # 简单深度图（前后一致，模拟平面站立）
    depth = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) * 180

    # 占位姿态图（实际使用时可替换为真实OpenPose生成的骨架图）
    pose = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    return pose, depth, mask_a, mask_b

def generate_handshake_conditions():
    """生成握手交互场景的掩码与深度图"""
    mask_a = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask_b = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    # 左右角色，中间手部区域重叠
    mask_a[:, :int(IMAGE_WIDTH*0.55)] = 255
    mask_b[:, int(IMAGE_WIDTH*0.45):] = 255

    depth = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8) * 180
    pose = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    return pose, depth, mask_a, mask_b

def generate_front_back_conditions():
    """生成前后站位场景的掩码与深度图"""
    mask_a = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask_b = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    # 前景角色（下方）、背景角色（上方）
    mask_a[int(IMAGE_HEIGHT*0.2):, :] = 255
    mask_b[:int(IMAGE_HEIGHT*0.8), :] = 255

    # 深度渐变，前近后远
    depth = np.linspace(255, 100, IMAGE_HEIGHT, dtype=np.uint8)
    depth = np.tile(depth[:, np.newaxis], (1, IMAGE_WIDTH))

    pose = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    return pose, depth, mask_a, mask_b

SCENE_GENERATORS = {
    "side_by_side": generate_side_by_side_conditions,
    "handshake": generate_handshake_conditions,
    "front_back": generate_front_back_conditions
}

def main():
    os.makedirs(CONDITION_DIR, exist_ok=True)

    for scene in SCENES:
        for bg in BACKGROUNDS:
            for (char_a, char_b) in CHARACTER_PAIRS:
                save_dir = os.path.join(CONDITION_DIR, scene, bg, f"{char_a}_{char_b}")
                os.makedirs(save_dir, exist_ok=True)

                pose, depth, mask_a, mask_b = SCENE_GENERATORS[scene]()

                cv2.imwrite(os.path.join(save_dir, "pose.png"), pose)
                cv2.imwrite(os.path.join(save_dir, "depth.png"), depth)
                cv2.imwrite(os.path.join(save_dir, "mask_a.png"), mask_a)
                cv2.imwrite(os.path.join(save_dir, "mask_b.png"), mask_b)
                print(f"Generated conditions: {save_dir}")

    print("All test conditions generated.")

if __name__ == "__main__":
    main()