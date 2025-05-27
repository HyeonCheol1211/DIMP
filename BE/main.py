from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: Optional[str] = None
    image_url: Optional[str] = None  # HttpUrl 대신 str로 변경

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    reply_image_url: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.image_url:
        # Base64 문자열이든 URL이든 문자열로 받음
        # 예시: 받은 이미지를 그대로 답장에 포함 (간단히)
        return ChatResponse(
            reply="이미지를 받았어요.",
            reply_image_url=req.image_url,
        )
    else:
        user_msg = req.message or ""
        return ChatResponse(reply=f"챗봇이 응답함: {user_msg}")

# AWS 검증용
@app.get("/health")
def health_check():
    return {"status": "healthy"}
