import cv2
import numpy as np

def visualize_mask(image_path, label_path):
    # 读取图像
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 创建空白mask
    mask = np.zeros((h, w), dtype=np.uint8)
    # 读取标签
    with open(label_path) as f:
        lines = f.readlines()
    # 绘制每个多边形
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        points = np.array(parts[1:]).reshape(-1, 2) * [w, h]
        points = points.astype(np.int32)
        cv2.fillPoly(mask, [points], color=class_id+1)  # +1避免0背景
    # 可视化
    cv2.imshow('Image', img)
    cv2.imshow('Mask', mask*30)  # 放大mask值便于观察
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例使用
# visualize_mask("data/images/Train/CCH_50F_op_frontal.jpg", "data/labels/Train/CCH_50F_op_frontal.txt")
