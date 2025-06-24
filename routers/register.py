from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import (
    User, get_db,Photo,
    insert_user, fetch_user_by_name, fetch_user_by_id, fetch_user_by_token, update_user, delete_user, send_email
)
import uuid
from datetime import datetime, timedelta
from enum import Enum
import jwt
import os
from dotenv import load_dotenv
import bcrypt

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
# 用户注册相关的API
class Gender(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"

class AgeGroup(str, Enum):
    Child = "6-12"  # 儿童
    Teen = "13-19"    # 成人
    Adult= "20-59"
    Senior = "60+"  # 老年人
    
class UserRegistration(BaseModel):
    gender: Gender
    age_group: AgeGroup
    username: str
    password: str
    email: str 
    phoneNumber: str
    fullName:str
    birthdate:str

    
#响应模型
class UserRegistrationResponse(BaseModel):
    userId: str
    message: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordReset(BaseModel):
    token: str
    new_password: str

router = APIRouter()
@router.post("/register", response_model=UserRegistrationResponse)
def register_user(user: UserRegistration, request: Request, db: Session = Depends(get_db)):
    # # Validate usertype
    # if user.usertype not in ["enduser", "reviewer"]:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Invalid usertype. Must be 'enduser' or 'reviewer'"
    #     )
    # Check for duplicate email
    existing_user = fetch_user_by_name(user.username, db)
    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Name already registered"
        )
    # 检查邮箱和手机号唯一性
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )
    if db.query(User).filter(User.phoneNumber == user.phoneNumber).first():
        raise HTTPException(
            status_code=409,
            detail="Phone number already registered"
        )
    # 密码加密
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # Create user record
    user_id = str(uuid.uuid4())
    user_data ={
        "userId":user_id,
        "username": user.username,
        "gender": user.gender,
        "email": user.email,
        "phoneNumber": user.phoneNumber,
        "fullName": user.fullName,
        "birthdate": user.birthdate, 
        "password": hashed_password,  # 存储加密后的密码
        "ageGroup": user.age_group,
        "createdAt": datetime.utcnow(),
        "lastUpdatedAt": datetime.utcnow(),
        "confirmed": False,
        "confirmed_at": None
    }
    insert_user(user_data, db)
    # 生成邮箱确认token
    confirm_token = jwt.encode({"userId": user_id, "exp": datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm=ALGORITHM)
    # 构造确认链接
    base_url = str(request.base_url).rstrip('/')
    confirm_url = f"{base_url}/v1/register/confirm/{confirm_token}"
    # 发送确认邮件
    subject = "Confirm your account"
    body = f"<p>Hi {user.username},</p><p>Please confirm your email by clicking the link below:</p><p><a href='{confirm_url}'>{confirm_url}</a></p>"
    send_email(user.email, subject, body)
    return UserRegistrationResponse(
        userId=user_id,
        message="Account created successfully. Please check your email to confirm your account."
    )

@router.get("/register/confirm/{token}")
def confirm_email(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        user = fetch_user_by_id(user_id, db)
        if not user:
            return {"message": "Invalid or expired token."}
        if user.confirmed:
            return {"message": "Account already confirmed."}
        update_user(user_id, {"confirmed": True, "confirmed_at": datetime.utcnow()}, db)
        return {"message": "Account confirmed successfully."}
    except Exception as e:
        return {"message": f"Invalid or expired token. {str(e)}"}

@router.post("/reset-password-request")
def reset_password_request(data: PasswordResetRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        # 为安全不暴露邮箱是否存在
        return {"message": "If this email is registered, a reset link has been sent."}
    reset_token = jwt.encode({"userId": user.userId, "exp": datetime.utcnow() + timedelta(hours=1)}, SECRET_KEY, algorithm=ALGORITHM)
    base_url = str(request.base_url).rstrip('/')
    reset_url = f"{base_url}/v1/register/reset-password?token={reset_token}"
    subject = "Reset your password"
    body = f"<p>Hi {user.username},</p><p>Click the link below to reset your password (valid for 1 hour):</p><p><a href='{reset_url}'>{reset_url}</a></p>"
    send_email(user.email, subject, body)
    return {"message": "If this email is registered, a reset link has been sent."}

@router.post("/reset-password")
def reset_password(data: PasswordReset, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(data.token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("userId")
        user = fetch_user_by_id(user_id, db)
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired token.")
        # 密码哈希加密存储
        hashed_pw = bcrypt.hashpw(data.new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        update_user(user_id, {"password": hashed_pw, "lastUpdatedAt": datetime.utcnow()}, db)
        return {"message": "Password reset successful."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or expired token. {str(e)}")