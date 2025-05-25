# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (React 개발서버에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 기본 개발 서버 주소
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
