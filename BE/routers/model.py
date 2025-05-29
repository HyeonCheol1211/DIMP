from fastapi import APIRouter
from transformers import AutoProcessor, AutoModelForImageTextToText, CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
from io import BytesIO
import base64
import torch
import faiss
import json
import requests

router = APIRouter()

# 모델 로드
model_id = "google/medgemma-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CLIP for vector DB
vdb_model_name = "openai/clip-vit-large-patch14-336"
vdb_model = CLIPModel.from_pretrained(vdb_model_name)
clip_processor = CLIPProcessor.from_pretrained(vdb_model_name)
vdb_model.eval()

# FAISS index 및 메타데이터 로드 (캐싱)
index = faiss.read_index("/home/ubuntu/DIMP/BE/skin_disease.index")
with open("/home/ubuntu/DIMP/BE/skin_disease_metadata.json", encoding="utf-8") as f:
    id_map = json.load(f)


# === 유틸 함수 ===

def load_image_from_input(image_input: str) -> Image.Image:
    # base64 이미지인지 확인
    if image_input.startswith("data:image"):
        header, encoded = image_input.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
    
    # http/https URL인 경우
    elif image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    
    # 로컬 파일 경로일 경우 (선택적으로 추가)
    elif image_input.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        image = Image.open(image_input)
    
    else:
        raise ValueError("지원하지 않는 이미지 형식입니다.")

    return image.convert("RGB")


def get_text_embedding(query_text: str):
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vec = vdb_model.get_text_features(**inputs)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec


def search_faiss(query_text, top_k=3):
    query_vec = get_text_embedding(query_text)
    D, I = index.search(query_vec.cpu().numpy(), top_k)
    results = []
    for rank, i in enumerate(I[0]):
        info = id_map[str(i)]
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "id": info["id"],
            "disease": info["disease"],
            "category": info["category"],
            "text": info["text"][:100] + "..."
        })
    return results


def generate_answer(query_text: str, image: Image.Image = None, context_texts: list = None):
    context = "\n".join([f"- {c}" for c in (context_texts or [])])
    prompt = f"""당신은 피부질환 전문가입니다. 아래 문맥을 참고하여 질문에 정확하게 답변하세요.

문맥:
{context}

질문:
{query_text}"""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_text},
                {"type": "image", "image": image}
            ]
        }
    ]

    # 프리프로세싱
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device, dtype=torch.bfloat16) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = output[0][input_len:]

    decoded = tokenizer.decode(generation, skip_special_tokens=True)
    return decoded


def multimodal_query(query_text=None, image=None, top_k=3):
    try:
        image_obj = load_image_from_input(image) if image else None
        search_results = search_faiss(query_text=query_text, top_k=top_k)
        context_texts = [r["text"] for r in search_results]
        return generate_answer(query_text, image_obj, context_texts)
    except Exception as e:
        return f"오류 발생: {str(e)}"
