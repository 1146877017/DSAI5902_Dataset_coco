import os
import cv2
import numpy as np
from ultralytics import YOLO

# 配置路径
raw_img_dir = r"coco_multi_person/raw"
pose_output_dir = r"coco_multi_person/openpose"
os.makedirs(pose_output_dir, exist_ok=True)

# 加载模型
model = YOLO("yolov8n-pose.pt")

# 按部位分组SKELETON + 定义部位颜色 
# 1. 按骨骼部位分组
SKELETON_BY_PART = {
    "head": [[0,1], [0,2], [1,2], [1,3], [2,4]],          # 头部
    "arm": [[5,6], [5,7], [6,8], [7,9], [8,10]],          # 手臂
    "torso": [[11,12], [5,11], [6,12]],                   # 躯干
    "leg": [[11,13], [12,14], [13,15], [14,16]]           # 腿
}

# 2. 定义各部位颜色（RGB格式）
PART_COLORS = {
    "head": (255, 0, 0),       # 头部 - 红色
    "arm": (0, 255, 0),        # 手臂 - 绿色
    "torso": (0, 0, 255),      # 躯干 - 蓝色
    "leg": (255, 255, 0)       # 腿部 - 黄色
}


for img_name in os.listdir(raw_img_dir):
    img_path = os.path.join(raw_img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    # 创建黑色画布
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 检测姿态
    results = model(img, conf=0.3)
    
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # 形状 
        for person_kps in keypoints:
            # 按部位绘制不同颜色线条 
            # 遍历每个部位，用对应颜色绘制
            for part_name, sk_list in SKELETON_BY_PART.items():
                color = PART_COLORS[part_name]  # 获取该部位的颜色
                for sk in sk_list:
                    pt1 = person_kps[sk[0]]
                    pt2 = person_kps[sk[1]]
                    # 检查关键点是否有效（非零坐标）
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, 2)
            
            # 绘制关键点（白色）
            for kp in person_kps:
                if kp[0] > 0 and kp[1] > 0:
                    cv2.circle(canvas, tuple(kp.astype(int)), 3, (255,255,255), -1)
    
    # 保存结果
    save_path = os.path.join(pose_output_dir, img_name)
    cv2.imwrite(save_path, canvas)

print("✅ 按骨骼部位分色的骨骼图生成完毕！")