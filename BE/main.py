from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from routers.model import multimodal_query

import base64
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: Optional[str] = None
    image_url: Optional[str] = None

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    reply_image_url: Optional[str] = None  # 향후 확장용

def load_image_from_input(image_input: str) -> Image.Image:
    try:
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(BytesIO(image_data)).convert("RGB")
        elif image_input.startswith("http://") or image_input.startswith("https://"):
            response = requests.get(image_input, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError("지원하지 않는 이미지 형식입니다.")
    except Exception as e:
        raise RuntimeError(f"이미지 로드 실패: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.image_url:
        return ChatResponse(reply="정확한 진단을 위해 이미지를 입력해주세요.")

    try:
        image = load_image_from_input(req.image_url)
    except Exception as e:
        return ChatResponse(reply=f"이미지 처리 실패: {str(e)}")

    try:
        outputs = multimodal_query(query_text=req.message or "", image=image)
        return ChatResponse(reply=outputs)
    except Exception as e:
        return ChatResponse(reply=f"멀티모달 응답 실패: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
