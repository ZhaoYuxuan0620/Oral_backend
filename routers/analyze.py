from fastapi import APIRouter, UploadFile, File, HTTPException, status, Path, Query, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import torch
import base64
import glob
from datetime import datetime
import cv2
router = APIRouter()

# 检查CUDA是否可用，设置device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载YOLOv8模型（只加载一次），指定device
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', '6_30best_aug.pt')
yolo_model = YOLO(MODEL_PATH)
yolo_model.to(DEVICE)

# 通道字典（如需后续多通道mask可参考）
# RED_DICT = {
#     0: "No teeth",
#     1: "Upper left central incisor (perm)",
#     2: "Upper right central incisor (perm)",
#     3: "Lower left central incisor (perm)",
#     # ...补充完整...
# }
# GREEN_DICT = {
#     0: "No issue",
#     1: "Moderate caries",
#     2: "Severe caries",
#     # ...补充完整...
# }
# BLUE_DICT = {
#     0: "No issue",
#     1: "Moderate gum inflammation",
#     2: "Severe gum inflammation",
#     # ...补充完整...
# }

def mask_channel_to_label_image(mask: np.ndarray, channel: int, label_dict: dict) -> Image:
    """
    将mask的某一通道转换为文字标注图
    """
    img = Image.fromarray(mask[..., channel], mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    h, w = mask.shape[:2]
    step = max(1, min(h, w) // 20)
    for y in range(0, h, step):
        for x in range(0, w, step):
            v = int(mask[y, x, channel])
            label = label_dict.get(v, str(v))
            draw.text((x, y), label, fill=(255, 0, 0), font=font)
    return img

@router.get("/analysis/mask/{userid}")
async def get_mask_image(
    userid: str,
    time: str = Query("latest", description="时间戳文件夹名，latest为最新，all为全部，none为按文件名"),
    filename: str = Query("none", description="mask文件名，仅当time=none时使用")
):
    user_dir = os.path.join("masks", userid)
    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail="User mask folder not found")
    # 获取所有时间戳文件夹
    subfolders = [f for f in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, f))]
    if not subfolders and time != "none":
        raise HTTPException(status_code=404, detail="No mask images found for user")
    if time == "latest":
        # 取最新时间戳文件夹
        latest_folder = max(subfolders, key=lambda x: x)
        mask_path = os.path.join(user_dir, latest_folder, "mask.png")
        if not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail="Mask image not found")
        return FileResponse(mask_path, media_type="image/png")
    elif time == "all":
        # 返回所有mask图片的文件路径
        mask_paths = []
        for folder in sorted(subfolders):
            mask_path = os.path.join(user_dir, folder, "mask.png")
            if os.path.exists(mask_path):
                mask_paths.append(f"{userid}/{folder}/mask.png")
        if not mask_paths:
            raise HTTPException(status_code=404, detail="No mask images found for user")
        return JSONResponse(content={"mask_paths": mask_paths})
    elif time == "none":
        # 通过filename读取图片
        if not filename:
            raise HTTPException(status_code=400, detail="filename required when time=none")
        mask_path = os.path.join("masks", filename)
        if not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail="Mask image not found")
        return FileResponse(mask_path, media_type="image/png")

@router.post(
    "/analysis",
    status_code=200,
    responses={
        400: {"description": "Invalid image format"},
        401: {"description": "Unauthorized"},
        404: {"description": "User not found"},
        500: {"description": "Processing error"}
    }
)
async def analyze_photos(
    image: UploadFile = File(..., description="Oral photo in JPEG/PNG format"),
    userId: str = Form(..., description="User ID")
):
    print(f"[DEBUG] analyze_photos received userid: {userId}")  # 添加这一行
    # 仅校验图像格式
    # if image.content_type not in ["image/jpeg", "image/png"]:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Invalid image format. Only JPEG/PNG accepted"
    #     )
    try:
        contents = await image.read()
        uploaded_img = Image.open(io.BytesIO(contents)).convert("RGB")
        # 使用YOLOv8模型推理，获取mask
        results = yolo_model(np.array(uploaded_img), device=DEVICE)
        # 合并所有牙齿的mask为一张类别分割图
        mask = None
        if hasattr(results[0], "masks") and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
            # 获取每个mask对应的类别ID
            if hasattr(results[0], "boxes") and hasattr(results[0].boxes, "cls"):
                class_ids = results[0].boxes.cls.cpu().numpy().astype(np.uint8)  # (N,)
            else:
                raise HTTPException(status_code=500, detail="Model did not return class ids")
            if masks.ndim == 3 and masks.shape[0] > 0 and masks.shape[0] == class_ids.shape[0]:
                max_probs = masks.max(axis=0)  # (H, W)
                mask_indices = masks.argmax(axis=0)  # (H, W), 取概率最大的mask索引
                # 构建类别ID图，背景为0
                mask = np.zeros_like(max_probs, dtype=np.uint8)
                for idx in range(masks.shape[0]):
                    mask[mask_indices == idx] = class_ids[idx] + 1  # 类别ID+1，0为背景
                mask[max_probs <= 0.5] = 0  # 阈值过滤，低于阈值为背景
            else:
                raise HTTPException(status_code=500, detail="Model did not return valid masks or class ids")
        else:
            raise HTTPException(status_code=500, detail="Model did not return a mask")
        # 类别英文标签
        class_labels = [
            "Background",    # 0
            "Tooth",         # 1
            "Caries",        # 2
            "GumInflam",     # 3
            "Mouth",         # 4
            "Gum",           # 5
            "Recession",     # 6
            "Ortho",         # 7
            "Mucosa",        # 8
            "Splint",        # 9
        ]
        h, w = mask.shape
        # 颜色渲染
        color_map = [
            (255, 255, 255),     # 0: Background (white)
            (255, 128, 0),       # 1: Tooth (vivid orange)
            (64, 0, 64),         # 2: Caries (dark purple)
            (255, 255, 0),       # 3: GumInflam (yellow)
            (0, 255, 255),       # 4: Mouth (cyan)
            (0, 255, 0),         # 5: Gum (green)
            (0, 128, 255),       # 6: Recession (blue-orange)
            (255, 0, 0),         # 7: Ortho (red)
            (128, 0, 255),       # 8: Mucosa (purple)
            (0, 0, 0),           # 9: Splint (black)
        ]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for v, color in enumerate(color_map):
            color_mask[mask == v] = color
        color_mask[(mask > len(color_map) - 1)] = (255, 255, 255)
        color_mask_pil = Image.fromarray(color_mask, mode="RGB").convert("RGBA")
        # 标注英文
        draw = ImageDraw.Draw(color_mask_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        mask_np = np.array(mask)
        # 取消标注英文的功能
        # class_labels = [
        #     "Tooth",           # 0
        #     "Caries",          # 1
        #     "Gum Inflammation",# 2
        #     "Mouth",           # 3
        #     "Gum",             # 4
        #     "Recession",       # 5
        #     "Orthodontics",    # 6
        #     "Mucosa",          # 7
        #     "Splint",          # 8
        # ]
        # from scipy.spatial.distance import cdist
        # for class_id in np.unique(mask_np):
        #     if class_id < 0 or class_id >= len(class_labels):
        #         continue
        #     binary = (mask_np == class_id).astype(np.uint8)
        #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     centers = []
        #     for cnt in contours:
        #         M = cv2.moments(cnt)
        #         if M["m00"] == 0:
        #             continue
        #         cx = int(M["m10"] / M["m00"])
        #         cy = int(M["m01"] / M["m00"])
        #         centers.append((cx, cy))
        #     # 合并距离较近的中心点
        #     merged = []
        #     used = set()
        #     threshold = 150  # 距离阈值，可调整
        #     for i, (cx, cy) in enumerate(centers):
        #         if i in used:
        #             continue
        #         group = [(cx, cy)]
        #         for j in range(i + 1, len(centers)):
        #             if j in used:
        #                 continue
        #             dist = np.sqrt((cx - centers[j][0]) ** 2 + (cy - centers[j][1]) ** 2)
        #             if dist < threshold:
        #                 group.append(centers[j])
        #                 used.add(j)
        #         used.add(i)
        #         # 取平均中心
        #         avg_cx = int(np.mean([pt[0] for pt in group]))
        #         avg_cy = int(np.mean([pt[1] for pt in group]))
        #         merged.append((avg_cx, avg_cy))
        #     label = class_labels[class_id]
        #     for (cx, cy) in merged:
        #         draw.text((cx, cy), label, fill=(255, 0, 0, 255), font=font)
        # 画轮廓线
        contour_img = np.array(color_mask_pil).copy()
        for class_id in np.unique(mask_np):
            if class_id == 0:
                continue
            binary = (mask_np == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (255, 255, 255, 255), 2)
        contour_mask_pil = Image.fromarray(contour_img).convert("RGBA")
        # 叠加到原图，增加透明度
        original_img = uploaded_img.resize(contour_mask_pil.size).convert("RGBA")
        # 增加透明度（更透明）
        alpha_mask = 100  # 0-255, 80更透明
        color_mask_pil.putalpha(alpha_mask)
        contour_mask_pil.putalpha(alpha_mask)
        # 先叠加颜色和文字，再叠加轮廓
        blended = Image.alpha_composite(original_img, color_mask_pil)
        blended = Image.alpha_composite(blended, contour_mask_pil)
        blended = blended.convert("RGB")
        # 判断userid是否为默认值
        if userId == "0":
            buf = io.BytesIO()
            blended.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        else:
            # 以时间戳为文件夹名
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            user_mask_dir = os.path.join("masks", userId, timestamp)
            os.makedirs(user_mask_dir, exist_ok=True)
            mask_path = os.path.join(user_mask_dir, "mask.png")
            blended.save(mask_path)
            return {
                "results": {
                    "maskFileName": f"{userId}/{timestamp}/mask.png"
                },
                "message": "Colored mask generated successfully"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Image processing error"
        )