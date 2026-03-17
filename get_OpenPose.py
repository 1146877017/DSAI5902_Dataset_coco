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

# COCO 格式的关键点连接关系
# 索引：0:鼻子,1:左眼,2:右眼,3:左耳,4:右耳,5:左肩,6:右肩,7:左肘,8:右肘,9:左腕,10:右腕,
#         11:左髋,12:右髋,13:左膝,14:右膝,15:左踝,16:右踝
SKELETON = [
    [0,1], [0,2], [1,2], [1,3], [2,4],          # 头部
    [5,6], [5,7], [6,8], [7,9], [8,10],          # 手臂
    [11,12], [5,11], [6,12], [11,13], [12,14],   # 躯干
    [13,15], [14,16]                              # 腿
]

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
        keypoints = results[0].keypoints.xy.cpu().numpy()  # 形状 [人数, 17, 2]
        for person_kps in keypoints:
            # 绘制连接线
            for sk in SKELETON:
                pt1 = person_kps[sk[0]]
                pt2 = person_kps[sk[1]]
                # 检查关键点是否有效（非零坐标）
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255,255,255), 2)
            # 绘制关键点
            for kp in person_kps:
                if kp[0] > 0 and kp[1] > 0:
                    cv2.circle(canvas, tuple(kp.astype(int)), 3, (255,255,255), -1)
    
    # 保存结果
    save_path = os.path.join(pose_output_dir, img_name)
    cv2.imwrite(save_path, canvas)

print("✅ 纯净的骨骼图（黑底白线）生成完毕！")