from fastapi import FastAPI, HTTPException, status, Depends, File, UploadFile, Form
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.sql import select
from sqlalchemy.orm import Session
import json
import os
import shutil
from PIL import Image
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

class PhotoMetadata(BaseModel):
    cameraModel: Optional[str] = None
    deviceModel: Optional[str] = None
    appVersion: Optional[str] = None

class PhotoUploadResponse(BaseModel):
    photoUrls: dict
    captureTimestamp: str
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

def validate_image(file: UploadFile) -> bool:
    try:
        # Read image to verify format and dimensions
        img = Image.open(file.file)
        width, height = img.size
        file.file.seek(0)  # Reset file pointer
        
        # Check dimensions
        if width < 1000 or height < 1000:
            return False
            
        # Check format
        if img.format.lower() not in ['jpeg', 'png']:
            return False
            
        # Check file size (10MB)
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset pointer
        if size > 10 * 1024 * 1024:
            return False
            
        return True
    except Exception:
        return False

def generate_signed_url(base_path: str, expiry: str, user_id: str) -> str:
    # TODO: Implement proper URL signing
    signature = "abcd1234"  # Replace with proper signing logic
    return f"https://api.myoral.ai{base_path}?expiry={expiry}&signature={signature}"


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

@app.post("/v1/upload/{userId}", response_model=PhotoUploadResponse)
async def upload_photos(
    userId: str,
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    captureTimestamp: str = Form(...),
    metadata: Optional[str] = Form(None),
    current_user_id: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # Permission check
    if userId != current_user_id:
        raise HTTPException(
            status_code=403,
            detail="Not allowed to upload photos for other users"
        )
    
    # Validate timestamp format
    try:
        capture_time = datetime.fromisoformat(captureTimestamp.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamp format. Use ISO 8601 format (e.g., 2025-04-16T12:30:00Z)"
        )

    # Validate metadata if provided
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            PhotoMetadata(**metadata_dict)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata format"
            )

    # Validate images
    photos = {"front": front, "left": left, "right": right}
    for name, photo in photos.items():
        if not validate_image(photo):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {name} image. Must be JPG/PNG, minimum 1000x1000px, under 10MB"
            )

    # Create upload directory
    timestamp_str = capture_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    upload_base = os.path.join("uploads", userId, timestamp_str)
    os.makedirs(upload_base, exist_ok=True)

    # Save files and generate URLs
    photo_urls = {}
    expiry = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    for photo_type, photo in photos.items():
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        extension = os.path.splitext(photo.filename)[1].lower() or ".jpg"
        filename = f"{file_id}-{photo_type}{extension}"
        file_path = os.path.join(upload_base, filename)
        
        # Save file
        with open(file_path, "wb+") as file_object:
            shutil.copyfileobj(photo.file, file_object)
        
        # Generate URL
        relative_path = f"/uploads/{userId}/{timestamp_str}/{filename}"
        photo_urls[photo_type] = generate_signed_url(relative_path, expiry, userId)

    return PhotoUploadResponse(
        photoUrls=photo_urls,
        captureTimestamp=timestamp_str,
        message="Photos uploaded successfully"
    )

@app.get("/debug", status_code=status.HTTP_200_OK)
def debug(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {
        "message": "Debug endpoint",
        "users": [u.userId for u in users]
    }