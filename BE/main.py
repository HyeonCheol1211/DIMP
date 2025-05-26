# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:5173",  # 클라이언트 개발환경
    os.getenv("FRONTEND_DOMAIN"),  # 배포된 프론트엔드 도메인
    "http://localhost:3000" # React 기본 개발 서버 주소
]

# CORS 설정 (React 개발서버에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:3000",  # React 기본 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 간단한 예시 챗봇 로직 (입력 메시지 반복)
    user_msg = req.message
    reply = f"챗봇이 응답함: {user_msg}"
    return ChatResponse(reply=reply)

# AWS 검증용
@app.get("/health")
def health_check():
    return {"status": "healthy"}
