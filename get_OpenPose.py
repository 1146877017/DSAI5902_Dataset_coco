import os
import cv2
from controlnet_aux import OpenposeDetector
from tqdm import tqdm

# 路径配置
raw_img_dir = r"coco_multi_person/raw"
pose_output_dir = r"coco_multi_person/openpose"
os.makedirs(pose_output_dir, exist_ok=True)

# 加载 ControlNet 官方标准 OpenPose 检测器
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# 批量生成标准姿态图
img_files = [f for f in os.listdir(raw_img_dir) if f.endswith(('.jpg','.png'))]
for img_name in tqdm(img_files, desc="生成标准OpenPose"):
    img_path = os.path.join(raw_img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None: continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_img = pose_detector(img_rgb, hand_and_face=False)
    save_path = os.path.join(pose_output_dir, os.path.splitext(img_name)[0] + ".png")
    pose_img.save(save_path)
    # pose_img.save(os.path.join(pose_output_dir, img_name))

print(" 标准OpenPose生成完成！ControlNet可完美识别！")