from fastapi import APIRouter, UploadFile, File, HTTPException, status, Path
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import torch

router = APIRouter()

# 检查CUDA是否可用，设置device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载YOLOv8模型（只加载一次），指定device
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')
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

@router.get("/analysis/{user_id}/mask/{filename}")
async def get_mask_image(user_id: str, filename: str):
    file_path = os.path.join("masks", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/png")

@router.post(
    "/analysis/{user_id}",
    status_code=200,
    responses={
        400: {"description": "Invalid image format"},
        401: {"description": "Unauthorized"},
        404: {"description": "User not found"},
        500: {"description": "Processing error"}
    }
)
async def analyze_photos(
    user_id: str = Path(..., description="User identifier"),
    image: UploadFile = File(..., description="Oral photo in JPEG/PNG format"),
):
    # 仅校验图像格式
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG/PNG accepted"
        )
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
        # 生成彩色可视化图
        # 颜色字典：0为背景，1-15为牙齿类别
        color_map = [
            (0, 0, 0),         # 0: 黑色（背景）
            (255, 0, 0),       # 1: 红色
            (0, 255, 0),       # 2: 绿色
            (0, 0, 255),       # 3: 蓝色
            (255, 255, 0),     # 4: 黄色
            (255, 0, 255),     # 5: 品红
            (0, 255, 255),     # 6: 青色
            (128, 128, 128),   # 7: 灰色
            (255, 128, 0),     # 8: 橙色
            (128, 0, 255),     # 9: 紫色
            (0, 128, 255),     # 10: 天蓝
            (128, 255, 0),     # 11: 黄绿
            (255, 0, 128),     # 12: 粉红
            (0, 255, 128),     # 13: 青绿
            (128, 0, 0),       # 14: 深红
            (0, 128, 0),       # 15: 深绿
        ]
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for v, color in enumerate(color_map):
            color_mask[mask == v] = color
        # 若类别ID大于color_map范围，统一为白色
        color_mask[(mask > len(color_map) - 1)] = (255, 255, 255)
        color_mask_pil = Image.fromarray(color_mask, mode="RGB")
        os.makedirs("./masks", exist_ok=True)
        mask_fn = f"{uuid.uuid4().hex}-colormask.png"
        color_mask_pil.save(f"./masks/{mask_fn}")
        base_url = "http://127.0.0.1:8000/v1/analysis"
        return {
            "results": {
                "colorMaskImageUrl": f"{base_url}/{user_id}/mask/{mask_fn}",
            },
            "message": "Colored mask generated successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Image processing error"
        )