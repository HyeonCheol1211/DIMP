from transformers import AutoProcessor
import requests
import torch
from llava.conversation import conv_templates, SeparatorStyle
from transformers import CLIPImageProcessor
from llava.model import *
from llava.mm_utils import KeywordsStoppingCriteria
from llava.model.multimodal_encoder.builder import build_vision_tower
from io import BytesIO
from PIL import Image
from llava.mm_utils import tokenizer_image_token

import faiss
import json
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from fastapi import APIRouter
router = APIRouter()

device = "cuda:0"
model_name = "tabtoyou/KoLLaVA-v1.5-Synatra-7b"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_TOKEN_INDEX = -200

tokenizer = AutoProcessor.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")
model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).to(device)
image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

vdb_model_name = "openai/clip-vit-large-patch14-336"
vdb_model = CLIPModel.from_pretrained(vdb_model_name)
#processor = CLIPProcessor.from_pretrained(vdb_model_name, torch_dtype=torch.float16)
vdb_model.eval()

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

from transformers import CLIPVisionModel
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
save_path = "vision_tower_finetuned_hf"
model.config.mm_vision_tower = save_path
model.get_model().vision_tower = build_vision_tower(model.config, delay_load=False).to(device)

vision_tower = model.get_model().vision_tower
vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

def get_text_embedding(query_text):
    inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vec = vdb_model.get_text_features(**inputs)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec



def search_faiss(query_text=None, image_path=None, top_k=3):
    if query_text is None and image_path is None:
        raise ValueError("텍스트 또는 이미지 입력이 하나 이상 필요합니다.")

    # 임베딩 생성
    embeddings = []
    if query_text:
        embeddings.append(get_text_embedding(query_text))
    if image_path:
        embeddings.append(get_image_embedding(image_path))

    # 평균 벡터 생성 (텍스트+이미지 둘 다 있는 경우)
    query_vec = torch.stack(embeddings).mean(dim=0)
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=1)

    # FAISS 검색
    index = faiss.read_index("skin_disease.index")
    with open("skin_disease_metadata.json", encoding="utf-8") as f:
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


def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vec = vdb_model.get_image_features(**inputs)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


tokenizer = AutoProcessor.from_pretrained("tabtoyou/KoLLaVA-v1.5-Synatra-7b")
model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).to(device)
image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

model.get_model().vision_tower = build_vision_tower(model.config, delay_load=False).to(device) # delayed load가 default로 되어 있어 외부에서 직접 정의
vision_tower = model.get_model().vision_tower
vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2


def inference(qs, image_file) :
    image = load_image(image_file)
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mistral"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str, "###"]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(device),
            images=image_tensor.unsqueeze(0).half().to(device),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria]).cpu()

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:-1], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs

def generate_answer_with_kollava(
    query_text: str,
    image_path: str = None,
    context_texts: list = None
):

    context = "\n".join([f"- {c}" for c in (context_texts or [])])
    qs = f"""당신은 피부질환 전문가입니다. 아래 문맥을 참고하여 질문에 정확하게 답변하세요.

문맥:
{context}

질문:
{query_text}"""

    image = load_image(image_path)
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mistral"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str, "###"]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to(device),
            images=image_tensor.unsqueeze(0).half().to(device),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria]).cpu()

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:-1], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def multimodal_kollava_query(query_text=None, image_path=None, top_k=3):
    search_results = search_faiss(query_text=query_text, image_path=image_path, top_k=top_k)
    context_texts = [r["text"] for r in search_results]
    return generate_answer_with_kollava(query_text, image_path, context_texts)