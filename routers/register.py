from fastapi import APIRouter, Depends, HTTPException, Request, Body
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
import requests

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID")
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
    gender: Gender = None
    age_group: AgeGroup = None
    username: str = None
    password: str = None
    email: str = None
    phoneNumber: str = None
    fullName: str = None
    birthdate: str = None
    register_method: str = "email"
    client_id: str = "0"  # 新增字段，默认为0
    
#响应模型
class UserRegistrationResponse(BaseModel):
    userId: str
    message: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordReset(BaseModel):
    token: str
    new_password: str

def format_hk_phone_number(phone: str) -> str:
    phone = phone.strip().replace(' ', '').replace('-', '')
    if phone.startswith('+852'):
        return phone
    if phone.startswith('852'):
        return '+' + phone
    if phone.startswith('+'):
        return phone
    return '+852' + phone

router = APIRouter()
@router.post("/register", response_model=UserRegistrationResponse)
def register_user(user: UserRegistration, request: Request, db: Session = Depends(get_db)):
    # 密码加密
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # Create user record
    user_id = str(uuid.uuid4())
    # user.phoneNumber = format_hk_phone_number(user.phoneNumber)
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
        "register_method": user.register_method,
        "confirmed": True,
        "confirmed_at": None,
        #"client_id": user.client_id if hasattr(user, 'client_id') else "0"
    }
    insert_user(user_data, db)
    return UserRegistrationResponse(userId=user_id, message="Account created.")