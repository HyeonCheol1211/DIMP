import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

from fastapi import APIRouter
router = APIRouter()

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id, torch_dtype=torch.float16, use_fast=True)

import faiss
import json
from transformers import CLIPProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor

vdb_model_name = "openai/clip-vit-base-patch32"
vdb_model = CLIPModel.from_pretrained(vdb_model_name, torch_dtype=torch.float16)
clip_processor = CLIPProcessor(
    tokenizer=CLIPTokenizer.from_pretrained(vdb_model_name, torch_dtype=torch.float16),
    image_processor=CLIPImageProcessor.from_pretrained(vdb_model_name, torch_dtype=torch.float16),
    torch_dtype=torch.float16, 
    use_fast=True)
vdb_model.eval()

def get_text_embedding(query_text):
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vec = vdb_model.get_text_features(**inputs)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec

def search_faiss(query_text, top_k=3):
    if not query_text:
        raise ValueError("텍스트 입력이 필요합니다.")

    # 텍스트 임베딩 생성
    query_vec = get_text_embedding(query_text)
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1)

    # FAISS 검색
    index = faiss.read_index("/home/ubuntu/DIMP/BE/skin_disease.index")
    with open("/home/ubuntu/DIMP/BE/skin_disease_metadata.json", encoding="utf-8") as f:
        id_map = json.load(f)

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

def generate_answer_text_only(query_text: str, context_texts: list = None):
    context = "\n".join([f"- {c}" for c in (context_texts or [])])
    prompt = f"""당신은 피부질환 전문가입니다. 아래 문맥을 참고하여 질문에 정확하게 답변하세요. 또한, 더욱 정확한 답변을 위해 이미지 첨부를 권장하여 주세요.

문맥:
{context}

질문:
{query_text}"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "text", "text": f"질문: {query_text}"}]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True)

def generate_answer_image_only(image: str):
    prompt = "당신은 피부질환 전문가입니다. 아래 이미지를 보고 질병에 대해 설명하세요. 또한, 더욱 정확한 답변을 위해 텍스트 입력을 권장하여 주세요."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [{"type": "image", "image": image}]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True)

def generate_answer_text_and_image(query_text: str, image: str, context_texts: list = None):
    context = "\n".join([f"- {c}" for c in (context_texts or [])])
    prompt = f"""당신은 피부질환 전문가입니다. 아래 문맥과 이미지를 참고하여 질문에 정확하게 답변하세요.

문맥:
{context}

질문:
{query_text}"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"질문: {query_text}"},
            {"type": "image", "image": image}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True)


def generate_answer(
    query_text: str = None,
    image: str = None,
    context_texts: list = None
):

    if query_text and image:
        return generate_answer_text_and_image(query_text, image, context_texts)
    elif query_text:
        return generate_answer_text_only(query_text, context_texts)
    elif image:
        return generate_answer_image_only(image)



def multimodal_query(query_text=None, image=None, top_k=3):
    if query_text is None and image is not None:
        image = image.resize((224, 224))
        return generate_answer(image=image)
    
    elif query_text is not None and image is None:
        search_results = search_faiss(query_text=query_text, top_k=top_k)
        context_texts = [r["text"] for r in search_results]
        return generate_answer(query_text=query_text, context_texts=context_texts)
    
    elif query_text is not None and image is not None:
        image = image.resize((224, 224))
        search_results = search_faiss(query_text=query_text, top_k=top_k)
        context_texts = [r["text"] for r in search_results]
        return generate_answer(query_text=query_text, image=image, context_texts=context_texts)
    
    else:
        raise ValueError("At least one of query_text or image must be provided.")
