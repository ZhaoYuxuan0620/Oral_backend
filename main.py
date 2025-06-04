from fastapi import FastAPI, HTTPException, status , Depends
from pydantic import BaseModel
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.sql import select
from sqlalchemy.orm import Session
from database import (
    User, get_db,
    insert_user, fetch_user_by_email, fetch_user_by_phone, fetch_user_by_id, fetch_user_by_token, update_user, delete_user
)

# FastAPI应用
app = FastAPI()
 
# request模型
class UserRegistration(BaseModel):
    email: str
    phoneNumber: str
    password: str
    usertype: str  # 'enduser' 或 'reviewer'
 
class UserLogin(BaseModel):
    userId: str = ""
    email: str = ""
    phoneNumber: str = ""
    password: str
 
# 响应模型
class UserRegistrationResponse(BaseModel):
    userId: str
    message: str
 
class UserLoginResponse(BaseModel):
    userId: str
    usertype: str
    token: str
    message: str

class UserDetailResponse(BaseModel):
    userId: str
    fullName: str
    email: str
    phoneNumber: str
    usertype: str
    createdAt: str
    lastUpdatedAt: str

class UpdateUserInfo(BaseModel):
    surname: Optional[str] = None
    givename: Optional[str] = None
    sex: Optional[str] = None
    birthYear: Optional[int] = None

class UserInfoResponse(BaseModel):
    surname: Optional[str]
    givename: Optional[str]
    sex: Optional[str]
    birthYear: Optional[int]
    message: str

#token verification
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    user = fetch_user_by_token(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return user.userId

@app.post("/v1/users/register",  status_code=status.HTTP_200_OK)
def register_user(user: UserRegistration, db: Session = Depends(get_db)):
    # Validate usertype
    if user.usertype not in ["enduser", "reviewer"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid usertype. Must be 'enduser' or 'reviewer'"
        )
    # Check for duplicate email
    existing_user = fetch_user_by_email(user.email, db)
    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )
    # Check for duplicate phone
    existing_phone = fetch_user_by_phone(user.phoneNumber, db)
    if existing_phone:
        raise HTTPException(
            status_code=409,
            detail="Phone number already registered"
        )
    # Create user record
    user_id = str(uuid.uuid4())
    user_data = {
        "userId": user_id,
        "email": user.email,
        "phoneNumber": user.phoneNumber,
        "password": user.password,  # Should hash in production
        "usertype": user.usertype,
        "createdAt": datetime.utcnow(),
        "lastUpdatedAt": datetime.utcnow(),
    }
    insert_user(user_data, db)
    return UserRegistrationResponse(
        userId=user_id,
        message="Account created successfully"
    )
 
@app.post("/v1/users/login",  status_code=status.HTTP_200_OK)
def login_user(login: UserLogin, db: Session = Depends(get_db)):
    user = None
    # Try to find user by userId, email, or phoneNumber
    if login.userId:
        user = fetch_user_by_id(login.userId, db)
    elif login.email:
        user = fetch_user_by_email(login.email, db)
    elif login.phoneNumber:
        user = fetch_user_by_phone(login.phoneNumber, db)
    # User not found
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
    # Password check
    if user.password != login.password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
    # Generate token
    token = str(uuid.uuid4())
    update_user(user.userId, {"token": token, "lastUpdatedAt": datetime.utcnow()}, db)
    return UserLoginResponse(
        userId=user.userId,
        usertype=user.usertype,
        token=token,
        message="Login successful"
    )
 
@app.put("/v1/users/{userId}", status_code=status.HTTP_200_OK)
def update_user_info(
    userId: str,
    update_data: UpdateUserInfo,
    current_user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Permission check
    if userId != current_user_id:
        raise HTTPException(
            status_code=403,
            detail="Not allowed to modify other users"
        )
    user = fetch_user_by_id(userId, db)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    # Validate sex
    if update_data.sex and update_data.sex not in ["M", "F"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid sex value. Accepted values: M, F"
        )
    # Prepare update fields
    update_fields = {k: v for k, v in update_data.dict().items() if v is not None}
    if update_fields:
        update_fields["lastUpdatedAt"] = datetime.utcnow()
        update_user(userId, update_fields, db)
    # Fetch updated user
    updated_user = fetch_user_by_id(userId, db)
    response_data = {
        "surname": updated_user.surname,
        "givename": updated_user.givename,
        "sex": updated_user.sex,
        "birthYear": updated_user.birthYear,
        "message": "User information updated successfully"
    }
    return response_data

# 获取用户信息
@app.get("/api/users/{userId}", response_model=UserDetailResponse, status_code=200)
def get_user_info(
    userId: str,
    current_user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Validate userId format (UUID)
    try:
        uuid.UUID(userId)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid userId format")
    # Find user
    user = fetch_user_by_id(userId, db)
    if not user:
        raise HTTPException(status_code=404, detail="User with the specified userId does not exist.")
    # Access control: allow self or reviewer
    current_user = fetch_user_by_id(current_user_id, db)
    if current_user_id != userId and (not current_user or current_user.usertype != "reviewer"):
        raise HTTPException(status_code=403, detail="Not allowed to view other users' info")
    # Compose full name
    full_name = f"{user.surname or ''} {user.givename or ''}".strip()
    # Use ISO format for createdAt/lastUpdatedAt, fallback to now if not present
    created_at = user.createdAt.isoformat() + "Z" if user.createdAt else datetime.utcnow().isoformat() + "Z"
    last_updated_at = user.lastUpdatedAt.isoformat() + "Z" if user.lastUpdatedAt else created_at
    return UserDetailResponse(
        userId=user.userId,
        fullName=full_name,
        email=user.email,
        phoneNumber=user.phoneNumber,
        usertype=user.usertype,
        createdAt=created_at,
        lastUpdatedAt=last_updated_at
    )

@app.get("/debug", status_code=status.HTTP_200_OK)
def debug(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {
        "message": "Debug endpoint",
        "users": [u.userId for u in users]
    }