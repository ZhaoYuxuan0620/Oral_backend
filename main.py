from fastapi import FastAPI, HTTPException, status ,Header, Depends
from pydantic import BaseModel
import uuid
from datetime import datetime
from typing import Optional, List
app = FastAPI()
 
# 内存模拟数据库
# 使用postgreSQL或MongoDB时
fake_db = []
 
# 请求模型
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
@app.post("/v1/users/register",  status_code=status.HTTP_200_OK)
async def register_user(user: UserRegistration):
    # 简单验证用户类型
    if user.usertype  not in ["enduser", "reviewer"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid usertype. Must be 'enduser' or 'reviewer'"
        )
   
    # 检查重复注册（简单示例）
    for existing_user in fake_db:
        if existing_user["email"] == user.email:
            raise HTTPException(
                status_code=409,
                detail="Email already registered"
            )
   
    # 创建用户记录
    user_id = str(uuid.uuid4())
    user_data = {
        "userId": user_id,
        "email": user.email,
        "phoneNumber": user.phoneNumber,
        "password": user.password,   # 实际应用中应使用密码哈希
        "usertype": user.usertype  
    }
    fake_db.append(user_data)
   
    return UserRegistrationResponse(
        userId=user_id,
        message="Account created successfully"
    )
 
@app.post("/v1/users/login",  status_code=status.HTTP_200_OK)
async def login_user(login: UserLogin):
    user = None
    # 通过三种方式查找用户
    for existing_user in fake_db:
        if (login.userId  and existing_user["userId"] == login.userId)  or \
           (login.email  and existing_user["email"] == login.email)  or \
           (login.phoneNumber  and existing_user["phoneNumber"] == login.phoneNumber):
            user = existing_user
            break
    # 用户不存在
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
    # 密码验证（简单字符串比较）
    if user["password"] != login.password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )
    # 生成简单token（实际应用中应使用JWT等）
    token = str(uuid.uuid4())
    user["token"] = token  # 保存token到用户数据
    return UserLoginResponse(
        userId=user["userId"],
        usertype=user["usertype"],
        token=token,
        message="Login successful"
    )
 
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
# 依赖函数：token验证
# def verify_token(authorization: str = Header(...)):
#     if not authorization.startswith("Bearer  "):
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid authentication scheme"
#         )
#     token = authorization.split("  ")[1]
   
#     # 简单检查token有效性（生产环境应使用JWT）
#     for user in fake_db:
#         if "token" in user and user["token"] == token:
#             return user["userId"]
   
#     raise HTTPException(
#         status_code=401,
#         detail="Invalid token"
#     )
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
 
security = HTTPBearer()
 
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # 简单检查token有效性
    for user in fake_db:
        if "token" in user and user["token"] == token:
            return user["userId"]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token"
    )
 
@app.put("/v1/users/{userId}", status_code=status.HTTP_200_OK)
async def update_user_info(
    userId: str,
    update_data: UpdateUserInfo,
    current_user_id: str = Depends(verify_token)
):
    # 检查用户权限
    if userId != current_user_id:
        raise HTTPException(
            status_code=403,
            detail="Not allowed to modify other users"
        )
    # 查找用户
    user = None
    for u in fake_db:
        if u["userId"] == userId:
            user = u    
            break
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    # 验证性别字段
    if update_data.sex and update_data.sex not in ["M", "F"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid sex value. Accepted values: M, F"
        )
    # 更新用户信息（只更新提供的字段）
    if update_data.surname:
        user["surname"] = update_data.surname  
    if update_data.givename:
        user["givename"] = update_data.givename  
    if update_data.sex:
        user["sex"] = update_data.sex  
    if update_data.birthYear:
        user["birthYear"] = update_data.birthYear  
    # 构建响应数据
    response_data = {
        "surname": user.get("surname"),
        "givename": user.get("givename"),
        "sex": user.get("sex"),
        "birthYear": user.get("birthYear"),
        "message": "User information updated successfully"
    }
    return response_data
# 获取用户信息
@app.get("/api/users/{userId}", response_model=UserDetailResponse, status_code=200)
async def get_user_info(
    userId: str,
    current_user_id: str = Depends(verify_token)
):
    # Validate userId format (UUID)
    try:
        uuid.UUID(userId)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid userId format")
 
    # Find user
    user = None
    for u in fake_db:
        if u["userId"] == userId:
            user = u
            break
    if not user:
        raise HTTPException(status_code=404, detail="User with the specified userId does not exist.")
 
    # Access control: allow self or reviewer
    # Find current user info
    current_user = None
    for u in fake_db:
        if u["userId"] == current_user_id:
            current_user = u
            break
    if current_user_id != userId and (not current_user or current_user.get("usertype") != "reviewer"):
        raise HTTPException(status_code=403, detail="Not allowed to view other users' info")
 
    # Compose full name
    full_name = f"{user.get('surname', '')} {user.get('givename', '')}".strip()
    # Use ISO format for createdAt/lastUpdatedAt, fallback to now if not present
    created_at = user.get("createdAt") or datetime.utcnow().isoformat() + "Z"
    last_updated_at = user.get("lastUpdatedAt") or created_at
 
    return UserDetailResponse(
        userId=user["userId"],
        fullName=full_name,
        email=user["email"],
        phoneNumber=user["phoneNumber"],
        usertype=user["usertype"],
        createdAt=created_at,
        lastUpdatedAt=last_updated_at
    )
@app.get("/debug", status_code=status.HTTP_200_OK)
async def debug():
    return {
        "message": "Debug endpoint",
        "fake_db": fake_db
    }