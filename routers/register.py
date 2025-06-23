from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import (
    User, get_db,Photo,
    insert_user, fetch_user_by_name, fetch_user_by_id, fetch_user_by_token, update_user, delete_user
)
import uuid
from datetime import datetime
from enum import Enum
import bcrypt  # 新增
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

router = APIRouter()
@router.post("/register", response_model=UserRegistrationResponse)
def register_user(user: UserRegistration, db: Session = Depends(get_db)):
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
    }
    insert_user(user_data, db)
    return UserRegistrationResponse(
        userId=user_id,
        message="Account created successfully"
    )