from fastapi import APIRouter, UploadFile, File, HTTPException, status, Path
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os
import uuid

router = APIRouter()

# 加载YOLOv8模型（只加载一次）
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_teeth_yolov8.pt')
yolo_model = YOLO(MODEL_PATH)

def yolo_detect(image: Image) -> Image:
    """
    使用YOLOv8模型检测牙齿,返回检测结果图像。
    """
    img_array = np.array(image.convert("RGB"))
    results = yolo_model(img_array)
    result_img = results[0].plot() if hasattr(results[0], 'plot') else img_array
    return Image.fromarray(result_img)

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
        detected_img = yolo_detect(uploaded_img)
        os.makedirs("./masks", exist_ok=True)
        detected_filename = f"{uuid.uuid4().hex}-detected.png"
        detected_img.save(f"./masks/{detected_filename}")
        detected_url = f"http://127.0.0.1:8000/v1/analysis/{user_id}/mask/{detected_filename}"
        return {
            "results": {
                "detectedImageUrl": detected_url
            },
            "message": "Teeth detection completed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Image processing error"
        )