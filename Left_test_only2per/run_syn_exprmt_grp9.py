import os
import json
import torch
import cv2
import numpy as np
import re
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.models.attention_processor import AttnProcessor

# ===================== 全局配置 =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
OUTPUT_DIR = "synthetic_results"          # 
LORA_WEIGHTS_DIR = "./lora_weights"
IMAGE_SIZE = 512
SEED = 42
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
generator = torch.Generator(device).manual_seed(SEED)

# ===================== 加载模型 =====================
print("=" * 60)
print(" 正在并行加载 ControlNet 模型与计算流...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# ===================== LoRA 动态加载 =====================
def load_loras_for_pair(pipe, char1_name, char2_name):
    """动态卸载旧权重并加载双角色并行兼容的 LoRA 权重体系"""
    pipe.unload_lora_weights()
    lora_path1 = os.path.join(LORA_WEIGHTS_DIR, f"{char1_name}.safetensors")
    lora_path2 = os.path.join(LORA_WEIGHTS_DIR, f"{char2_name}.safetensors")
    if not os.path.exists(lora_path1) or not os.path.exists(lora_path2):
        raise FileNotFoundError(f"LoRA 文件夹权重缺失: '{lora_path1}' 或 '{lora_path2}'")
    
    pipe.load_lora_weights(lora_path1, adapter_name="char1")
    pipe.load_lora_weights(lora_path2, adapter_name="char2")
    pipe.set_adapters(["char1", "char2"], adapter_weights=[0.8, 0.8])

# ===================== Token 定位函数 =====================
def get_person_token_indices(tokenizer, prompt):
    """确保百分之百准确定位 person1 和 person2 的语义区间，杜绝越界错误"""
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    p1_idx = [i for i, t in enumerate(clean_tokens) if "person1" in t]
    p2_idx = [i for i, t in enumerate(clean_tokens) if "person2" in t]
    
    if not p1_idx:
        p1_idx = [i for i, t in enumerate(clean_tokens) if t == "person" and i+1 < len(clean_tokens) and "1" in clean_tokens[i+1]]
    if not p2_idx:
        p2_idx = [i for i, t in enumerate(clean_tokens) if t == "person" and i+1 < len(clean_tokens) and "2" in clean_tokens[i+1]]
        
    if not p1_idx or not p2_idx:
        print(f"[!] 警告: 文本标记定位失败，启用标准 fallback 区间")
        return [list(range(5, 15)), list(range(16, 26))]
        
    start_p1 = p1_idx[0]
    start_p2 = p2_idx[0]
    
    stop_tokens = [",", ".", "and", "with", "a", "an", "the", "in", "of"]
    
    person1_token_ids = [
        idx for idx in range(start_p1, start_p2) 
        if clean_tokens[idx] not in stop_tokens
    ]
    
    end_p2 = len(clean_tokens) - 1
    for i in range(start_p2, len(clean_tokens)):
        if any(keyword in clean_tokens[i] for keyword in ["scene", "outdoor", "indoor", "street", "<|endoftext|>", ".", ","]):
            end_p2 = i - 1
            break    
        
    person2_token_ids = [
        idx for idx in range(start_p2, end_p2 + 1) 
        if clean_tokens[idx] not in stop_tokens
    ]
    return [person1_token_ids, person2_token_ids]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

# ===================== 交叉注意力掩码拦截处理器 =====================
class AttentionMaskProcessor(object):
    def __init__(self, token_indices, masks, penalty_weight=15.0):
        super().__init__()
        self.token_indices = token_indices  
        self.masks = masks                  
        self.penalty_weight = penalty_weight 

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        is_self_attn = encoder_hidden_states is None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale

        if not is_self_attn and self.token_indices:
            spatial_seq_len = attn_scores.shape[-2]
            spatial_size = int(np.sqrt(spatial_seq_len))
            
            if spatial_size in [16, 32]:
                chunks = attn_scores.shape[0] // 2
                
                for i, token_ids in enumerate(self.token_indices):
                    if i >= len(self.masks):
                        break
                    
                    mask_img = self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.NEAREST)
                    m_arr = np.array(mask_img) / 255.0
                    mask_tensor = torch.tensor(m_arr, device=query.device, dtype=query.dtype)
                    
                    penalty_matrix = (1.0 - mask_tensor).view(-1, 1) * (-self.penalty_weight)
                    
                    for token_idx in token_ids:
                        if token_idx >= attn_scores.shape[-1]:
                            continue
                        attn_scores[chunks:, :, token_idx] += penalty_matrix.squeeze(-1)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attn.head_to_batch_dim(attention_mask)
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def apply_attention_mask(pipe, token_indices, masks, penalty_weight=15.0):
    pipe.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks, penalty_weight))

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 主流程（只处理第一组） =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if len(configs) == 0:
        print("没有找到任何样本配置，退出。")
        return

    # 只取第9个样本
    cfg = configs[8]
    sample_id = cfg["sample_id"]
    prompt = cfg["prompt"]
    neg_prompt = cfg["negative_prompt"]

    char_to_filename = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    char1_name = char_to_filename[cfg["characters"][0]]
    char2_name = char_to_filename[cfg["characters"][1]]

    prompt = re.sub(r"<lora:[^>]+>", "", prompt)
    print(f"\n正在处理第4组自动化流水线样本: {sample_id}")

    for pipe in [pipe_base, pipe_pose, pipe_both]:
        load_loras_for_pair(pipe, char1_name, char2_name)

    pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
    depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
    mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
    
    token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

    # 1. Baseline 1
    img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
    img1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[0]}.png"))
    clear_gpu_memory()

    # 2. Baseline 2
    img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose,
                     controlnet_conditioning_scale=0.6, generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[1]}.png"))
    clear_gpu_memory()

    # 3. Baseline 3
    img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                     controlnet_conditioning_scale=[0.6, 0.5], generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img3.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[2]}.png"))
    clear_gpu_memory()

    # 4. Method
    masks = process_mask(mask_path)
    apply_attention_mask(pipe_both, token_indices, masks, penalty_weight=15.0)
    img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                     controlnet_conditioning_scale=[0.6, 0.5], generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
    pipe_both.unet.set_attn_processor(AttnProcessor())
    clear_gpu_memory()

    print(f" [+] 第5组样本 {sample_id} 的 4 个对比实验数据全部落地完成。")
    print(f"\n实验完成，结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        print(f"\n运行时遇到致命阻断错误: {str(e)}")
        clear_gpu_memory()
        raise