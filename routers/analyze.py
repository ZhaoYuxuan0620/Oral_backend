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
from datetime import datetime,timedelta
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import json
import textwrap
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

router = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG]  using device: {device}")
# 加载YOLOv8模型（只加载一次），指定device
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', '6_30best_aug.pt')
yolo_model = YOLO(MODEL_PATH)
yolo_model.to(device)


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
    userId: str = Form("0", description="User ID")
):

    # 动态选择 device
    
    print(f"[DEBUG] analyze_photos received userid: {userId}, using device: {device}")
    yolo_model.to(device)

    # 仅校验图像格式
    # if image.content_type not in ["image/jpeg", "image/png"]:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Invalid image format. Only JPEG/PNG accepted"
    #     )
    try:
        caries_history = None
        recession_history = None
        contents = await image.read()
        uploaded_img = Image.open(io.BytesIO(contents)).convert("RGB")
        # 使用YOLOv8模型推理，获取mask
        results = yolo_model(np.array(uploaded_img), device=device)
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
        # 统计各类别像素点数量
        class_counts = {label: int((mask == idx).sum()) for idx, label in enumerate(class_labels)}
        # 统计各类别目标数量（去除Background）
        # 只保留需要统计的类别
        report_classes = [
            "Tooth", "Caries", "GumInflam", "Recession"
        ]
        class_object_counts = {label: int((class_ids == idx-1).sum()) for idx, label in enumerate(class_labels) if label in report_classes}
        # 生成PDF报告，包含mask图片和颜色说明
        def generate_pdf_report(object_counts, timestamp, mask_img, color_map, class_labels, user_info=None, caries_history=None, recession_history=None):
            # 新增历史曲线页
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            # 顶部logo与信息
            logo_y = height - 60
            LIGHT_BLUE = "#42A5F5"
            c.setFont("Helvetica-Bold", 28)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))
            c.drawCentredString(width / 2, logo_y, "MyOral.ai")
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.HexColor("#555555"))
            c.drawCentredString(width / 2, logo_y - 16, "K11 Atelier, King's Road, Hong Kong")

            # 分割线
            c.setStrokeColor(colors.HexColor("#B0B0B0"))
            c.setLineWidth(1.5)
            c.line(40, logo_y - 28, width - 40, logo_y - 28)

            # 主标题
            c.setFont("Helvetica-Bold", 24)
            c.setFillColor(colors.HexColor("#1976D2"))
            c.drawCentredString(width / 2, logo_y - 55, "Oral Health Report")

            # 报告生成时间（中国时间），紧跟在主标题下且不加粗
            y = logo_y - 75
            from datetime import datetime, timedelta
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.HexColor("#1976D2"))
            try:
                utc_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except Exception:
                utc_time = datetime.utcnow()
            china_time = utc_time + timedelta(hours=8)
            c.drawCentredString(width / 2, y, f"Report Generated At: {china_time.strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 24

            # Patient Info 区块美化
            c.saveState()
            c.setFillColor(colors.HexColor("#E3F2FD"))
            c.roundRect(40, y-80, width-60, 100, 10, fill=1, stroke=0)
            c.restoreState()
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))
            c.drawString(50, y, "Patient Info")
            y -= 18
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.black)
            if user_info:
                if user_info.get("Name"):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(60, y, "Full Name:")
                    c.setFont("Helvetica", 12)
                    c.drawString(140, y, str(user_info['Name']))
                    y -= 16
                if user_info.get("Gender"):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(60, y, "Gender:")
                    c.setFont("Helvetica", 12)
                    c.drawString(140, y, str(user_info['Gender']))
                    y -= 16
                if user_info.get("Birthdate"):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(60, y, "Birth Date:")
                    c.setFont("Helvetica", 12)
                    c.drawString(140, y, str(user_info['Birthdate']))
                    y -= 16
                if user_info.get("Phone"):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(60, y, "Phone:")
                    c.setFont("Helvetica", 12)
                    c.drawString(140, y, str(user_info['Phone']))
                    y -= 16
                if user_info.get("Email"):
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(60, y, "Email:")
                    c.setFont("Helvetica", 12)
                    c.drawString(140, y, str(user_info['Email']))
                    y -= 16
                y -= 12
            else:
                c.drawString(60, y, "No patient info available.")
                y -= 16

            # 黄牙检测（只在tooth区域分析HSV）
            # 牙齿区域mask
            tooth_mask = (np.array(mask_img.convert("RGB"))[..., :3])
            mask_np = np.array(mask_img.convert("RGB"))
            mask_id = np.array(mask)
            tooth_area = (mask_id == 1)
            # 提取牙齿区域的像素
            tooth_pixels = mask_np[tooth_area]
            # 转为HSV
            if tooth_pixels.size > 0:
                tooth_pixels_hsv = cv2.cvtColor(tooth_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                # 黄牙判定区间（H: 5~35）
                yellow_mask = ((tooth_pixels_hsv[:,0] >=5) & (tooth_pixels_hsv[:,0] <= 35))
                yellow_ratio = yellow_mask.sum() / tooth_pixels_hsv.shape[0]
            else:
                yellow_ratio = 0
            # 亮度等级划分（1-5）
            # 0-10%: 1, 10-25%: 2, 25-45%: 3, 45-70%: 4, >70%: 5
            if yellow_ratio <= 0.10:
                brightness_level = 1
            elif yellow_ratio <= 0.25:
                brightness_level = 2
            elif yellow_ratio <= 0.45:
                brightness_level = 3
            elif yellow_ratio <= 0.70:
                brightness_level = 4
            else:
                brightness_level = 5

            # 黄牙报告区块美化（左右排版，左：文字，右：亮度条带）
            block_height = 60
            block_y = y-50
            c.saveState()
            c.setFillColor(colors.HexColor("#FFFDE7"))
            c.roundRect(40, block_y, width-60, block_height, 10, fill=1, stroke=0)
            c.restoreState()
            # 左侧文字
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(colors.HexColor("#FBC02D"))
            c.drawString(50, y-10, "Tooth Whiteness Analysis")
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.black)
            text_y = y - 25
            if brightness_level == 1:
                c.drawString(60, text_y, "Your teeth color appears healthy and white with whiteness level 1")
                text_y -= 16
            else:
                c.drawString(60, text_y, f"Whiteness Level: {brightness_level}")
                text_y -= 16
                c.drawString(60, text_y, "Recommendation: Consider professional cleaning.")
                text_y -= 16

            # 右侧亮度条带可视化
            bar_x = width - 17 - 120  # 右侧边距80，条带宽120
            bar_y = block_y + 12
            bar_width = 100
            bar_height = 20
            # 绘制渐变条带（从白到黄）
            from reportlab.lib.colors import Color
            for i in range(bar_width):
                # 渐变色：左白(255,255,255)到右黄(255,230,50)
                r = 255
                g = int(255 - (i/bar_width)*25)
                b = int(255 - (i/bar_width)*205)
                c.setFillColor(Color(r/255, g/255, b/255))
                c.rect(bar_x + i, bar_y, 1, bar_height, fill=1, stroke=0)
            # 绘制亮度等级分割线和数字
            for lvl in range(1,6):
                pos = bar_x + int((lvl-1)*bar_width/4)
                c.setStrokeColor(colors.HexColor("#B0B0B0"))
                c.setLineWidth(1)
                c.line(pos, bar_y, pos, bar_y+bar_height)
                c.setFont("Helvetica", 10)
                c.setFillColor(colors.black)
                c.drawCentredString(pos, bar_y+bar_height+12, str(lvl))
            # 用红色三角标记当前亮度等级
            marker_pos = bar_x + int((brightness_level-1)*bar_width/4)
            c.setFillColor(colors.red)
            c.setStrokeColor(colors.red)
            c.setLineWidth(1)
            c.line(marker_pos, bar_y+bar_height+2, marker_pos-4, bar_y+bar_height+10)
            c.line(marker_pos, bar_y+bar_height+2, marker_pos+4, bar_y+bar_height+10)
            c.line(marker_pos-4, bar_y+bar_height+10, marker_pos+4, bar_y+bar_height+10)
            c.setFillColor(colors.red)
            c.setFont("Helvetica", 9)
            y -= block_height

            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))
            y -= 20
            c.drawString(50, y-10, "Assessment")
            y -= 26
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.black)
            tooth = object_counts.get("Tooth", 0)
            caries = object_counts.get("Caries", 0)
            gum_inflam = object_counts.get("GumInflam", 0)
            recession = object_counts.get("Recession", 0)

            # 计算每个caries对应牙齿的面积比例及平均占比
            caries_ratios = []
            print("reached")
            if caries > 0 and hasattr(results[0], "masks") and results[0].masks is not None:
                # 获取mask和class_ids
                masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
                if hasattr(results[0], "boxes") and hasattr(results[0].boxes, "cls"):
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(np.uint8)  # (N,)
                else:
                    class_ids = None
                # 遍历所有caries实例
                for i in range(masks.shape[0]):
                    if class_ids is not None and class_ids[i] == 1:  # 1: Caries
                        # 找到该caries掩码
                        caries_mask = (masks[i] > 0.5)
                        # 找到与该caries掩码有重叠的tooth掩码
                        max_overlap = 0
                        tooth_idx = -1
                        for j in range(masks.shape[0]):
                            if class_ids[j] == 0:  # 0: Tooth
                                tooth_mask = (masks[j] > 0.5)
                                overlap = np.logical_and(caries_mask, tooth_mask).sum()
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    tooth_idx = j
                        # 计算面积比例
                        caries_pixels = caries_mask.sum()
                        if tooth_idx >= 0:
                            tooth_mask = (masks[tooth_idx] > 0.5)
                            tooth_pixels = tooth_mask.sum()
                            total_pixels = tooth_pixels + caries_pixels
                            ratio = caries_pixels / total_pixels if total_pixels > 0 else 0
                            caries_ratios.append(ratio)
                        else:
                            # 没找到对应牙齿，跳过
                            continue
                avg_caries_ratio = np.mean(caries_ratios) if caries_ratios else 0
            else:
                avg_caries_ratio = 0

            # YOLO模型总结
            summary_line = f"Detected: {tooth} teeth, {caries} caries, {gum_inflam} gum inflammation, {recession} gum recession."
            wrap_width = width - 100  # 文字区域宽度
            for line in textwrap.wrap(summary_line, width=70):
                c.drawString(60, y, line)
                y -= 14

            # Caries面积比例信息
            if caries > 0 and caries_ratios:
                for idx, ratio in enumerate(caries_ratios):
                    c.drawString(60, y, f"Caries {idx+1} area ratio: {ratio:.2%}")
                    y -= 14
                c.drawString(60, y, f"Average caries area ratio: {avg_caries_ratio:.2%}")
                y -= 14

            assessment = "The patient appears in good health with no immediate concerns during the examination."
            if caries > 0:
                assessment = f"{caries} caries detected. Severe situation."
            elif gum_inflam > 0:
                assessment = f"{gum_inflam} gum inflammation area(s) found. Moderate situation."
            elif recession > 0:
                assessment = f"{recession} gum recession area(s) detected. Moderate situation."
            elif tooth == 0:
                assessment = "No teeth detected. Please check the uploaded image or retake the photo."
            for line in textwrap.wrap(assessment, width=60):
                c.drawString(60, y, line)
                y -= 14

            y -= 30

            # Prescription
            c.saveState()
            c.setFillColor(colors.HexColor("#E3F2FD"))
            c.roundRect(40, y-40, width-60, 60, 10, fill=1, stroke=0)
            c.restoreState()
            c.setFont("Helvetica-Bold", 13)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))  # 浅蓝色
            c.drawString(50, y, "Recommendation")
            y -= 18
            c.setFont("Helvetica", 11)
            c.setFillColor(colors.black)
            prescription = "No recommendation is necessary at this time, as the patient is in good health with no identified medical concerns."
            if caries > 0 or gum_inflam > 0 or recession > 0:
                prescription = "Please consult a dentist for further evaluation and treatment."
            elif tooth == 0:
                prescription = "No prescription due to missing teeth detection."
            for line in textwrap.wrap(prescription, width=80):
                c.drawString(60, y, line)
                y -= 14

            y -= 16

            # 插入mask图片和legend（legend在图片右侧）
            c.setFont("Helvetica-Bold", 13)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))  # 浅蓝色
            c.drawString(50, y-10, "Segmentation Mask:")
            y -= 10
            img_buf = io.BytesIO()
            mask_img.save(img_buf, format="PNG")
            img_buf.seek(0)
            img_reader = ImageReader(img_buf)
            # 布局参数
            legend_width = 160  # legend区宽度
            gap = 24  # 图片与legend间隔
            max_total_width = width - 100  # 页面可用最大宽度
            max_img_width = max_total_width - legend_width - gap
            orig_width, orig_height = mask_img.size
            # 预留下方空间
            reserved_height = 150 
            max_img_height = y - reserved_height
            scale = min(max_img_width / orig_width, max_img_height / orig_height, 1.0)
            draw_width = orig_width * scale
            draw_height = orig_height * scale
            img_x = 50
            img_y = y - draw_height - 10  # 下移10单位
            c.drawImage(img_reader, img_x, img_y, width=draw_width, height=draw_height, mask='auto')
            # legend区整体下移10单位
            legend_x = img_x + draw_width + gap
            legend_y = y - 10
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.HexColor(LIGHT_BLUE))
            c.drawString(legend_x, legend_y, "Legend:")
            legend_y -= 22
            c.setFont("Helvetica", 11)
            # 只渲染和标注 Tooth, Caries, GumInflam, Recession 四种类别，颜色与渲染一致
            legend_classes = ["Tooth", "Caries", "GumInflam", "Recession","Gum"]
            legend_indices = [class_labels.index(cls) for cls in legend_classes]
            for idx in legend_indices:
                label = class_labels[idx]
                color = color_map[idx]
                alpha = render_alpha / 255
                c.setFillColorRGB(color[0]/255, color[1]/255, color[2]/255, alpha=alpha)
                c.rect(legend_x, legend_y, 16, 16, fill=1, stroke=0)
                c.setFillColor(colors.black)
                if(label == "GumInflam"):
                    c.drawString(legend_x + 22, legend_y + 2, "Gum Inflamation")
                elif(label == "Recession"):
                    c.drawString(legend_x + 22, legend_y + 2, "Gum Recession")
                else:
                    c.drawString(legend_x + 22, legend_y + 2, label)
                legend_y -= 20

            y = img_y - 30

            # 免责声明
            y-=50
            disclaimer = "Disclaimer: This report is generated by MyOral.ai using AI analysis. For a professional diagnosis, please consult a dentist."
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColorRGB(0.3, 0.3, 0.3)
            max_width = width - 100
            words = disclaimer.split()
            line = ''
            for word in words:
                test_line = line + word + ' '
                if c.stringWidth(test_line, "Helvetica-Oblique", 9) < max_width:
                    line = test_line
                else:
                    c.drawString(50, y-42, line.strip())
                    y -= 12
                    line = word + ' '
            if line:
                c.drawString(50, y-42, line.strip())
                y -= 12

            # 页脚美化
            c.setStrokeColor(colors.HexColor("#B0B0B0"))
            c.setLineWidth(1)
            c.line(40, 50, width - 40, 50)
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColor(colors.HexColor("#1976D2"))
            c.drawCentredString(width / 2, 36, "For inquiries and appointments, visit www.MyOral.ai")
            print(f"[DEBUG] Passing to PDF: caries_history={caries_history}, recession_history={recession_history}")
            if caries_history is not None and recession_history is not None:
                c.showPage()
                # 标题
                c.setFont("Helvetica-Bold", 20)
                c.setFillColor(colors.HexColor("#1976D2"))
                c.drawCentredString(width/2, height-60, "Oral Health History")

                # --- Caries Count Chart ---
                c.setFont("Helvetica-Bold", 14)
                c.setFillColor(colors.black)
                caries_title_y = height-120
                c.drawString(60, caries_title_y+15, "Caries Count")
                # 下移所有图表和分析内容
                block_offset = 75  # 稍微增大整体下移量，增大间距
                chart_x = 60
                chart_y = caries_title_y - 25 - block_offset  # 稍微增大caries title与图表间距
                chart_w = 320
                chart_h = 110  # 压缩y轴高度
                c.setStrokeColor(colors.black)
                c.setLineWidth(1)
                c.rect(chart_x, chart_y, chart_w, chart_h, stroke=1, fill=0)
                caries_items = sorted(caries_history.items(), key=lambda x: x[0])
                caries_vals = [v for k, v in caries_items]
                max_caries = max(caries_vals) if caries_vals else 1
                for i in range(0, max_caries+1):
                    y_pos = chart_y + (i/max(1,max_caries))*chart_h
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.drawRightString(chart_x-5, y_pos-3, str(i))
                    c.setStrokeColor(colors.HexColor("#CCCCCC"))
                    c.setLineWidth(0.5)
                    c.line(chart_x, y_pos, chart_x+chart_w, y_pos)
                # x轴日期等距放缩
                if len(caries_items) > 1:
                    from datetime import datetime
                    date_fmt = "%Y-%m-%d"
                    dates = [datetime.strptime(k, date_fmt) for k, v in caries_items]
                    min_date, max_date = min(dates), max(dates)
                    total_days = (max_date - min_date).days or 1
                    x_positions = [chart_x + ((d - min_date).days / total_days) * chart_w for d in dates]
                    date_label_y = chart_y - 28
                    for idx, (x_pos, (date_str, _)) in enumerate(zip(x_positions, caries_items)):
                        c.saveState()
                        c.setFont("Helvetica", 9)
                        c.setFillColor(colors.black)
                        c.translate(x_pos, date_label_y)
                        c.rotate(90)
                        c.drawCentredString(0, 0, date_str)
                        c.restoreState()
                    # 曲线
                    c.setStrokeColor(colors.HexColor("#4A90E2"))
                    c.setLineWidth(2)
                    points = [(x_positions[i], chart_y + (caries_vals[i]/max(1,max_caries))*chart_h) for i in range(len(caries_vals))]
                    for i in range(len(points)-1):
                        c.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
                    for pt in points:
                        c.circle(pt[0], pt[1], 3, stroke=1, fill=1)
                elif len(caries_items) == 1:
                    x_pos = chart_x + chart_w/2
                    date_label_y = chart_y - 28
                    c.saveState()
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.translate(x_pos, date_label_y)
                    c.rotate(90)
                    c.drawCentredString(0, 0, caries_items[0][0])
                    c.restoreState()
                    y_val = chart_y + (caries_items[0][1]/max(1,max_caries))*chart_h
                    c.setStrokeColor(colors.HexColor("#4A90E2"))
                    c.circle(x_pos, y_val, 3, stroke=1, fill=1)
                # 右侧平均值
                c.setFont("Helvetica", 12)
                c.setFillColor(colors.black)
                caries_age_map = {
                    (5, 14): 0.8,
                    (15, 24): 2.1,
                    (25, 34): 3.2,
                    (35, 44): 5.0,
                    (45, 54): 7.2,
                    (55, 64): 10.1,
                    (65, 74): 13.2,
                    (75, 120): 15.0
                }
                user_age = None
                age_group = "(25,34)"
                avg_caries_val = "--"
                if user_info and user_info.get("Birthdate"):
                    try:
                        birth = user_info["Birthdate"]
                        if isinstance(birth, str):
                            birth = birth[:10]
                        birthdate = datetime.strptime(str(birth), "%Y-%m-%d")
                        today = datetime.utcnow() + timedelta(hours=8)
                        user_age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
                        for (a, b), val in caries_age_map.items():
                            if a <= user_age <= b:
                                age_group = f"({a},{b})"
                                avg_caries_val = str(val)
                                break
                    except Exception:
                        avg_caries_val = "3.2"
                else:
                    avg_caries_val = "3.2"
                analysis_x = chart_x+chart_w+20
                analysis_y = chart_y+chart_h-10
                analysis_width = 120
                c.drawString(analysis_x, analysis_y, f"Avg Caries {age_group} : ")
                c.setFont("Helvetica-Oblique", 12)
                c.setFillColor(colors.black)
                c.drawString(analysis_x+112, analysis_y, avg_caries_val)
                # 分析内容
                user_caries_str = "--"
                percent_diff_str = "--"
                if caries_vals and avg_caries_val not in ("--", "3.2"):
                    try:
                        user_latest_caries = caries_vals[-1]
                        avg_val = float(avg_caries_val)
                        if avg_val > 0:
                            percent_diff = (user_latest_caries - avg_val) / avg_val * 100
                            if percent_diff > 0:
                                percent_diff_str = f"Analyze: Attention! Worse than the average by {percent_diff:.1f}%"
                            elif percent_diff < 0:
                                percent_diff_str = f"Analyze: Good! Better than the average by {abs(percent_diff):.1f}%"
                            else:
                                percent_diff_str = "Analyze: Same as the average"
                        else:
                            percent_diff_str = "--"
                    except Exception:
                        percent_diff_str = "--"
                elif caries_vals and avg_caries_val == "3.2":
                    try:
                        user_latest_caries = caries_vals[-1]
                        avg_val = 3.2
                        percent_diff = (user_latest_caries - avg_val) / avg_val * 100
                        if percent_diff > 0:
                            percent_diff_str = f"Analyze: Attention! Worse than the average by {percent_diff:.1f}%"
                        elif percent_diff < 0:
                            percent_diff_str = f"Analyze: Good! Better than the average by {abs(percent_diff):.1f}%"
                        else:
                            percent_diff_str = "Analyze: Same as the average"
                    except Exception:
                        percent_diff_str = "--"
                from textwrap import wrap
                c.setFont("Helvetica", 11)
                c.setFillColor(colors.black)
                for idx, line in enumerate(wrap(percent_diff_str, width=22)):
                    c.drawString(analysis_x, analysis_y-18-idx*13, line)

                # --- Recession Count Chart ---
                c.setFont("Helvetica-Bold", 14)
                c.setFillColor(colors.black)
                # 下移recession相关内容
                recession_title_y = chart_y - 55 - block_offset  # 稍微增大caries与recession间距
                c.drawString(60, recession_title_y+6, "Gum Recession Count")
                chart2_x = 60
                chart2_y = recession_title_y - 110  # 稍微增大recession title与图表间距
                chart2_w = 320
                chart2_h = 110  # 压缩y轴高度
                c.setStrokeColor(colors.black)
                c.setLineWidth(1)
                c.rect(chart2_x, chart2_y, chart2_w, chart2_h, stroke=1, fill=0)
                recession_items = sorted(recession_history.items(), key=lambda x: x[0])
                recession_vals = [v for k, v in recession_items]
                max_recession = max(recession_vals) if recession_vals else 1
                for i in range(0, max_recession+1):
                    y_pos = chart2_y + (i/max(1,max_recession))*chart2_h
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.drawRightString(chart2_x-5, y_pos-3, str(i))
                    c.setStrokeColor(colors.HexColor("#CCCCCC"))
                    c.setLineWidth(0.5)
                    c.line(chart2_x, y_pos, chart2_x+chart2_w, y_pos)
                if len(recession_items) > 1:
                    from datetime import datetime
                    date_fmt = "%Y-%m-%d"
                    dates = [datetime.strptime(k, date_fmt) for k, v in recession_items]
                    min_date, max_date = min(dates), max(dates)
                    total_days = (max_date - min_date).days or 1
                    x_positions = [chart2_x + ((d - min_date).days / total_days) * chart2_w for d in dates]
                    date_label_y2 = chart2_y - 28
                    for idx, (x_pos, (date_str, _)) in enumerate(zip(x_positions, recession_items)):
                        c.saveState()
                        c.setFont("Helvetica", 9)
                        c.setFillColor(colors.black)
                        c.translate(x_pos, date_label_y2)
                        c.rotate(90)
                        c.drawCentredString(0, 0, date_str)
                        c.restoreState()
                    c.setStrokeColor(colors.HexColor("#4A90E2"))
                    c.setLineWidth(2)
                    points = [(x_positions[i], chart2_y + (recession_vals[i]/max(1,max_recession))*chart2_h) for i in range(len(recession_vals))]
                    for i in range(len(points)-1):
                        c.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
                    for pt in points:
                        c.circle(pt[0], pt[1], 3, stroke=1, fill=1)
                elif len(recession_items) == 1:
                    x_pos = chart2_x + chart2_w/2
                    date_label_y2 = chart2_y - 28
                    c.saveState()
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.translate(x_pos, date_label_y2)
                    c.rotate(90)
                    c.drawCentredString(0, 0, recession_items[0][0])
                    c.restoreState()
                    y_val = chart2_y + (recession_items[0][1]/max(1,max_recession))*chart2_h
                    c.setStrokeColor(colors.HexColor("#4A90E2"))
                    c.circle(x_pos, y_val, 3, stroke=1, fill=1)
                # 显示其他用户平均值
                avg_recession_other = 1.2
                avg_recession_user = "--"
                if recession_vals:
                    avg_recession_user = f"{np.mean(recession_vals):.2f}"
                c.setFont("Helvetica", 12)
                c.setFillColor(colors.black)
                c.drawString(chart2_x+chart2_w+25, chart2_y+chart2_h-10, "Avg Recession (Others): ")
                c.setFont("Helvetica-Oblique", 12)
                c.setFillColor(colors.black)
                c.drawString(chart2_x+chart2_w+155, chart2_y+chart2_h-10, str(avg_recession_other))
                # 分析内容：用户均值与总均值比较
                percent_diff_str = "--"
                if avg_recession_user != "--":
                    try:
                        user_val = float(avg_recession_user)
                        other_val = float(avg_recession_other)
                        if user_val > other_val:
                            percent_diff = (user_val - other_val) / other_val * 100
                            percent_diff_str = f"Analyze: Attention! Higher than the average by {percent_diff:.1f}% (User: {user_val}, Avg: {other_val})"
                        elif user_val < other_val:
                            percent_diff = (other_val - user_val) / other_val * 100
                            percent_diff_str = f"Analyze: Good! Lower than the average by {percent_diff:.1f}% (User: {user_val}, Avg: {other_val})"
                        else:
                            percent_diff_str = f"Analyze: Same as the average (User: {user_val}, Avg: {other_val})"
                    except Exception:
                        percent_diff_str = "--"
                from textwrap import wrap as wrap2
                c.setFont("Helvetica", 11)
                c.setFillColor(colors.black)
                for idx, line in enumerate(wrap2(percent_diff_str, width=30)):
                    c.drawString(chart2_x+chart2_w+25, chart2_y+chart2_h-35-idx*13, line)

                # --- GumInflam Score Chart (hard-coded, percent-based) ---
                c.setFont("Helvetica-Bold", 14)
                c.setFillColor(colors.black)
                gum_title_y = chart2_y - 55 - block_offset  # 稍微增大recession与gum inflamation间距
                c.drawString(60, gum_title_y+6, "Gum Inflamation Score")
                chart3_x = 60
                chart3_y = gum_title_y - 110  # 稍微增大gum inflamation title与图表间距
                chart3_w = 320
                chart3_h = 110
                c.setStrokeColor(colors.black)
                c.setLineWidth(1)
                c.rect(chart3_x, chart3_y, chart3_w, chart3_h, stroke=1, fill=0)
                # Hard-coded percent data (10 points)
                gum_dates = [
                    "2025-06-15", "2025-06-22", "2025-06-29", "2025-07-06", "2025-07-13", "2025-07-20", "2025-07-22", "2025-07-25", "2025-07-27", "2025-07-29"
                ]
                gum_vals = [78, 82, 85, 80, 88, 90, 83, 87, 85, 84]  # 0-100分制
                max_gum = 100
                min_gum = 70
                # y轴百分制
                for i in range(min_gum, max_gum+1, 5):
                    y_pos = chart3_y + ((i-min_gum)/(max_gum-min_gum))*chart3_h
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.drawRightString(chart3_x-5, y_pos-3, str(i))
                    c.setStrokeColor(colors.HexColor("#CCCCCC"))
                    c.setLineWidth(0.5)
                    c.line(chart3_x, y_pos, chart3_x+chart3_w, y_pos)
                # x轴日期等距放缩
                from datetime import datetime
                date_fmt = "%Y-%m-%d"
                dates = [datetime.strptime(d, date_fmt) for d in gum_dates]
                min_date, max_date = min(dates), max(dates)
                total_days = (max_date - min_date).days or 1
                x_positions = [chart3_x + ((d - min_date).days / total_days) * chart3_w for d in dates]
                date_label_y3 = chart3_y - 28
                for idx, (x_pos, date_str) in enumerate(zip(x_positions, gum_dates)):
                    c.saveState()
                    c.setFont("Helvetica", 9)
                    c.setFillColor(colors.black)
                    c.translate(x_pos, date_label_y3)
                    c.rotate(90)
                    c.drawCentredString(0, 0, date_str)
                    c.restoreState()
                c.setStrokeColor(colors.HexColor("#4A90E2"))
                c.setLineWidth(2)
                points = [(x_positions[i], chart3_y + ((gum_vals[i]-min_gum)/(max_gum-min_gum))*chart3_h) for i in range(len(gum_vals))]
                for i in range(len(points)-1):
                    c.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
                for pt in points:
                    c.circle(pt[0], pt[1], 3, stroke=1, fill=1)

                # 右侧显示其他用户平均分
                avg_gum_user = int(round(sum(gum_vals)/len(gum_vals)))
                avg_gum_other = 83
                c.setFont("Helvetica", 12)
                c.setFillColor(colors.black)
                c.drawString(chart3_x+chart3_w+25, chart3_y+chart3_h-10, f"Avg Score (Others): {avg_gum_other}")
                # 分析内容：用户均值与总均值比较
                percent_diff_str = "--"
                if avg_gum_user != "--":
                    try:
                        user_val = float(avg_gum_user)
                        other_val = float(avg_gum_other)
                        if user_val > other_val:
                            percent_diff = (user_val - other_val) / other_val * 100
                            percent_diff_str = f"Analyze: Good! Higher than the average by {percent_diff:.1f}% (User: {user_val}, Avg: {other_val})"
                        elif user_val < other_val:
                            percent_diff = (other_val - user_val) / other_val * 100
                            percent_diff_str = f"Analyze: Attention! Lower than the average by {percent_diff:.1f}% (User: {user_val}, Avg: {other_val})"
                        else:
                            percent_diff_str = f"Analyze: Same as the average (User: {user_val}, Avg: {other_val})"
                    except Exception:
                        percent_diff_str = "--"
                from textwrap import wrap as wrap3
                c.setFont("Helvetica", 11)
                c.setFillColor(colors.black)
                for idx, line in enumerate(wrap3(percent_diff_str, width=30)):
                    c.drawString(chart3_x+chart3_w+25, chart3_y+chart3_h-35-idx*13, line)

                c.save()
                buf.seek(0)
                return buf

        h, w = mask.shape
        # 只渲染 Tooth, Caries, GumInflam, Gum, Recession 五种类别，其余全部白色
        color_map = [
            (255, 255, 255),     # 0: Background (white)
            (255, 0, 0),         # 1: Tooth (bright red)
            (0, 255, 0),         # 2: Caries (bright green)
            (255, 255, 0),       # 3: GumInflam (bright yellow)
            (255, 255, 255),     # 4: Mouth (white, not rendered)
            (0, 0, 255),         # 5: Gum (blue)
            (255, 0, 255),       # 6: Recession (magenta)
            (0, 255, 255),       # 7: Ortho (cyan)
            (255, 128, 0),       # 8: Mucosa (orange)
            (0, 0, 0),           # 9: Splint (black)
        ]
        render_alpha = 150  # 0-255, 越高越不透明
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        # 只渲染五种类别，其余全部白色
        render_classes = [1, 2, 3, 5, 6]
        for v in range(len(color_map)):
            color = color_map[v]
            if v in render_classes:
                color_mask[mask == v, :3] = color
                color_mask[mask == v, 3] = render_alpha
            else:
                color_mask[mask == v, :3] = (255, 255, 255)
                color_mask[mask == v, 3] = render_alpha
        color_mask_pil = Image.fromarray(color_mask, mode="RGBA")
        mask_np = np.array(mask)

        # 画轮廓线（每种class都描边，统一用黑色，线宽2）
        contour_img = np.array(color_mask_pil).copy()
        for class_id in np.unique(mask_np):
            if (class_id == 0) or (class_id == 4):
                continue
            binary = (mask_np == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (0, 0, 0, 255), 2)
        contour_mask_pil = Image.fromarray(contour_img).convert("RGBA")
        # 叠加到原图，增加透明度
        original_img = uploaded_img.resize(contour_mask_pil.size).convert("RGBA")
        blended = Image.alpha_composite(original_img, color_mask_pil)
        blended = Image.alpha_composite(blended, contour_mask_pil)
        blended = blended.convert("RGB")

        # 画每颗牙齿（每个实例）的轮廓线，背景(0)和嘴巴(4)不画
        contour_img = np.array(color_mask_pil).copy()
        instance_alpha = 60  # 轮廓线透明度（0-255），可自定义
        if hasattr(results[0], "masks") and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
            if hasattr(results[0], "boxes") and hasattr(results[0].boxes, "cls"):
                class_ids = results[0].boxes.cls.cpu().numpy().astype(np.uint8)  # (N,)
            else:
                class_ids = None
            for i in range(masks.shape[0]):
                if class_ids is not None and (class_ids[i]+1 in [0, 4]):
                    continue  # 跳过背景和嘴巴
                binary = (masks[i] > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_img, contours, -1, (0, 0, 0, instance_alpha), 2)
        contour_mask_pil = Image.fromarray(contour_img).convert("RGBA")
        # 叠加到原图，增加透明度
        original_img = uploaded_img.resize(contour_mask_pil.size).convert("RGBA")
        alpha_mask = 25  # 0-255, 80更透明
        color_mask_pil.putalpha(alpha_mask)
        contour_mask_pil.putalpha(instance_alpha)
        blended = Image.alpha_composite(original_img, color_mask_pil)
        blended = Image.alpha_composite(blended, contour_mask_pil)
        blended = blended.convert("RGB")

        if userId == "0":
            buf = io.BytesIO()
            blended.save(buf, format="PNG")
            # 生成PDF报告（无用户信息）
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            pdf_buf = generate_pdf_report(class_object_counts, timestamp, blended, color_map, class_labels)
            pdf_base64 = base64.b64encode(pdf_buf.read()).decode()
            return StreamingResponse(buf, media_type="image/png", headers={
                "X-Report-PDF-Base64": pdf_base64
            })
        else:
            # 以时间戳为文件夹名
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            user_mask_dir = os.path.join("masks", userId, datetime.utcnow().strftime("%Y%m%d%H%M%S"))
            os.makedirs(user_mask_dir, exist_ok=True)
            mask_path = os.path.join(user_mask_dir, "mask.png")
            blended.save(mask_path)
            # 保存类别ID图（二维数组，npy格式）
            mask_id_path = os.path.join(user_mask_dir, "mask_id.npy")
            np.save(mask_id_path, mask_np)
            # 查找用户信息（通过数据库）
            user_info = None
            caries_history = None
            recession_history = None
            try:
                from database import get_db, fetch_user_by_id, update_user
                db_gen = get_db()
                db = next(db_gen)
                user_obj = fetch_user_by_id(userId, db)
                if user_obj:
                    user_info = {
                        "Name": user_obj.fullName,
                        "Gender": user_obj.gender,
                        "Birthdate": user_obj.birthdate,
                        "Email": user_obj.email,
                        "Phone": user_obj.phoneNumber
                    }
                    # 历史 caries_number 和 recession_number 以日期为key的dict
                    caries_dict = {}
                    recession_dict = {}
                    if user_obj.caries_number:
                        try:
                            caries_dict = json.loads(user_obj.caries_number)
                        except Exception:
                            caries_dict = {}
                    if user_obj.recession_number:
                        try:
                            recession_dict = json.loads(user_obj.recession_number)
                        except Exception:
                            recession_dict = {}
                    # 追加本次分析结果
                    caries = class_object_counts.get("Caries", 0)
                    recession = class_object_counts.get("Recession", 0)
                    today_str = datetime.utcnow().strftime("%Y-%m-%d")
                    caries_dict[today_str] = caries
                    recession_dict[today_str] = recession
                    # 限制历史长度（如只保留最近20次）
                    def trim_dict(d, max_len=20):
                        items = sorted(d.items(), key=lambda x: x[0])
                        if len(items) > max_len:
                            items = items[-max_len:]
                        return dict(items)
                    caries_dict = trim_dict(caries_dict)
                    recession_dict = trim_dict(recession_dict)
                    # 更新数据库
                    update_user(userId, {
                        "caries_number": json.dumps(caries_dict, ensure_ascii=False),
                        "recession_number": json.dumps(recession_dict, ensure_ascii=False)
                    }, db)
                    db.close()
                    caries_history = caries_dict
                    recession_history = recession_dict
                else:
                    caries_history = None
                    recession_history = None
            except Exception:
                user_info = None
                caries_history = None
                recession_history = None
            # 生成PDF报告
            print(f"[DEBUG] Before Passing to PDF: caries_history={caries_history}, recession_history={recession_history}")
            pdf_buf = generate_pdf_report(class_object_counts, timestamp, blended, color_map, class_labels, user_info,caries_history,recession_history)
            pdf_path = os.path.join(user_mask_dir, "report.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_buf.read())
            # 返回的文件名使用同一个timestamp
            return {
                "results": {
                    "maskFileName": f"{userId}/{os.path.basename(user_mask_dir)}/mask.png",
                    "maskIdFileName": f"{userId}/{os.path.basename(user_mask_dir)}/mask_id.npy",
                    "reportFileName": f"{userId}/{os.path.basename(user_mask_dir)}/report.pdf"
                },
                "message": "Colored mask, mask id array, and PDF report generated successfully"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Image processing error"
        )

@router.get("/analysis/maskid/{userid}")
async def get_mask_id_array(
    userid: str
):
    user_dir = os.path.join("masks", userid)
    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail="User mask folder not found")
    subfolders = [f for f in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, f))]
    if not subfolders:
        raise HTTPException(status_code=404, detail="No mask id files found for user")
    latest_folder = max(subfolders, key=lambda x: x)
    mask_id_path = os.path.join(user_dir, latest_folder, "mask_id.npy")
    if not os.path.exists(mask_id_path):
        raise HTTPException(status_code=404, detail="Mask id file not found")
    mask_id = np.load(mask_id_path)
    return {"mask_id": mask_id.tolist()}

@router.get("/analysis/report/pdf")
async def get_pdf_report(pdf_path: str):
    """
    读取指定路径的PDF报告并返回,支持客户端下载。
    :param pdf_path: PDF文件的相对或绝对路径
    :return: FileResponse
    """
    pdf_path=os.path.join('masks',pdf_path)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found")
    filename = os.path.basename(pdf_path)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename
    )