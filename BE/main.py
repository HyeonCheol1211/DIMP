from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from routers.model import multimodal_query

import base64
from io import BytesIO
from PIL import Image
import httpx  # ✅ 비동기 요청용 라이브러리
import asyncio

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

async def load_image_from_input(image_input: str) -> Image.Image:
    try:
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(BytesIO(image_data)).convert("RGB")

        elif image_input.startswith("http://") or image_input.startswith("https://"):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(image_input)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"🔴 외부 이미지 요청 실패: {str(e)}")

        else:
            raise ValueError("지원하지 않는 이미지 형식입니다.")

    except Exception as e:
        raise RuntimeError(f"이미지 로드 실패: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    print(f"[요청 수신] message={req.message}, image_url={(req.image_url or '')[:30]}...")

    image = None  # 초기화 필수

    if req.image_url is not None and req.image_url.strip() != "":
        try:
            image = await load_image_from_input(req.image_url)
            print("✅ 이미지 로드 성공")
        except Exception as e:
            print(f"❌ 이미지 처리 실패: {e}")
            return ChatResponse(reply=f"이미지 처리 실패: {str(e)}", reply_image_url=None)

    try:
        outputs = multimodal_query(query_text=req.message, image=image)
        print("✅ 멀티모달 쿼리 성공")
        return ChatResponse(reply=outputs, reply_image_url=None)
    except Exception as e:
        print(f"❌ 멀티모달 실패: {e}")
        return ChatResponse(reply=f"멀티모달 응답 실패: {str(e)}", reply_image_url=None)
