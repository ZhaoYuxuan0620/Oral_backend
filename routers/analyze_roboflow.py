from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cv2
import numpy as np
import supervision as sv
import io
import inference
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    ROBOFLOW_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()

router = APIRouter()

# 设置Roboflow API key
os.environ["ROBOFLOW_API_KEY"] = settings.ROBOFLOW_API_KEY

# 初始化模型（全局变量，避免每次请求都重新加载）
MODEL = inference.get_roboflow_model("tooth_gums/2")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # 可在此处添加实际token校验逻辑
    return token

@router.post("/v1/analysis/{user_id}")
async def analyze_image(user_id: str, file: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    分析上传的牙齿图像
    """
    try:
        # 读取上传的图像
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 运行模型推理
        results = MODEL.infer(image)
        
        # 调试输出：打印结果结构
        print("Raw inference results:", results)
        
        # 尝试不同的结果解析方式
        try:
            # 尝试Roboflow API的标准输出格式
            detections = sv.Detections.from_roboflow(results)
        except Exception:
            try:
                # 尝试YOLOv8格式
                detections = sv.Detections.from_inference(results[0] if isinstance(results, list) else results)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to parse model output: {str(e)}. Output format: {type(results)}"
                )
        
        # 创建标注器
        box_annotator = sv.BoxAnnotator()
        
        # 在图像上添加标注
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        
        # 将标注后的图像转换为bytes
        is_success, buffer = cv2.imencode(".png", annotated_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        
        return StreamingResponse(
            io.BytesIO(buffer), 
            media_type="image/png"
        )
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))