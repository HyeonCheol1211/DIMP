from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

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
            reply="그렇군요. 불편하지는 않으셔서 다행이에요.\n보내주신 사진 상으로는 마치 닭살처럼 돌기가 많이 있는 모습이네요.\n이런경우 다음과 같은 질환일 가능성이 있어요.\n모공성 각화증\n긁거나 자극을 주면 염증이 생길 수 있으니 가능하면 자극을 주지 않는 것이 좋아요. 건강상에 문제는 없으니 걱정하지 않으셔도 돼요. 자외선 차단제를 바르면 각화증 방지에 도움이 돼요. 드물게 편평세포암으로 발전할 수 있어요.\n\n증상이 심하거나 장기간 지속된다면 피부과 전문의와의 상담이 필요해요.",
        )
    else:
        user_msg = req.message or ""
        return ChatResponse(reply=f"챗봇이 응답함: {user_msg}")
