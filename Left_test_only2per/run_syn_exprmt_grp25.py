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
OUTPUT_DIR = "synthetic_results"      
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
print(" 正在并行加载 OpenPose 官方彩色编码标准与计算流...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def load_loras_for_pair(pipe, char1_name, char2_name):
    pipe.unload_lora_weights()
    lora_path1 = os.path.join(LORA_WEIGHTS_DIR, f"{char1_name}.safetensors")
    lora_path2 = os.path.join(LORA_WEIGHTS_DIR, f"{char2_name}.safetensors")
    if not os.path.exists(lora_path1) or not os.path.exists(lora_path2):
        raise FileNotFoundError(f"LoRA 权重缺失: '{lora_path1}' 或 '{lora_path2}'")
    
    pipe.load_lora_weights(lora_path1, adapter_name="char1")
    pipe.load_lora_weights(lora_path2, adapter_name="char2")
    pipe.set_adapters(["char1", "char2"], adapter_weights=[0.8, 0.8])

def get_person_token_indices(tokenizer, prompt):
    """
    
    完全绕开文本分词不确定性，精准切分 person1 描述区间与 person2 描述区间
    """
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    try:
        p1_start = -1
        for i in range(len(clean_tokens) - 1):
            if clean_tokens[i] == "person" and "1" in clean_tokens[i+1]:
                p1_start = i
                break
        
        p2_start = -1
        for i in range(len(clean_tokens) - 1):
            if clean_tokens[i] == "person" and "2" in clean_tokens[i+1]:
                p2_start = i
                break
                
        scene_start = len(clean_tokens) - 1
        for i in range(max(0, p2_start), len(clean_tokens)):
            if any(k in clean_tokens[i] for k in ["scene", "side", "handshake", "front", "outdoor", "indoor", "street", "<|endoftext|>", ".", ","]):
                scene_start = i
                break
        
        if p1_start != -1 and p2_start != -1:
            # Person1 的控制范围涵盖到 Person2 出现前
            person1_token_ids = list(range(p1_start, p2_start))
            # Person2 的控制范围涵盖到场景/背景词出现前
            person2_token_ids = list(range(p2_start, scene_start))
            return [person1_token_ids, person2_token_ids]
    except Exception as e:
        print(f"[-] Token 边界动态解析异常，触发安全后备机制: {e}")
        
    return [list(range(5, 15)), list(range(16, 26))]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

# =====================  全分辨率交叉注意力拦截拦截器 =====================
class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks, penalty_weight=20.0):
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

        # 仅在交叉注意力层且目标分辨率上施加 token 级惩罚
        if not is_self_attn and self.token_indices:
            spatial_seq_len = attn_scores.shape[-2]
            spatial_size = int(np.sqrt(spatial_seq_len))
            if spatial_size in [8, 16, 32, 64]:
                chunks = attn_scores.shape[0] // 2
                for i, token_ids in enumerate(self.token_indices):
                    if i >= len(self.masks):
                        break
                    mask_img = self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.NEAREST)
                    m_arr = np.array(mask_img) / 255.0
                    mask_tensor = torch.tensor(m_arr, device=query.device, dtype=query.dtype)
                    penalty_vector = (1.0 - mask_tensor).view(-1) * (-self.penalty_weight)
                    for token_idx in token_ids:
                        if token_idx >= attn_scores.shape[-1]:
                            continue
                        attn_scores[chunks:, :, token_idx] += penalty_vector

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

def apply_attention_mask(pipe, token_indices, masks, penalty_weight=20.0):
    pipe.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks, penalty_weight))

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 主计算流程 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if len(configs) == 0:
        return

    # 精准锁定第25个样本做验证
    cfg = configs[24]
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
    print(f"\n正在执行学术控制拦截流水线，目标验证样本: {sample_id}")

    for pipe in [pipe_base, pipe_pose, pipe_both]:
        load_loras_for_pair(pipe, char1_name, char2_name)

    pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
    depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
    mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
    
    token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

    # 1. Baseline 1
    print(" >>> 正在生成 Baseline 1 (Base Pipeline)...")
    img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
    img1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[0]}.png"))
    clear_gpu_memory()

    # 2. Baseline 2
    print(" >>> 正在生成 Baseline 2 (Single ControlNet Pose)...")
    img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose,
                     controlnet_conditioning_scale=0.7, generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[1]}.png"))
    clear_gpu_memory()

    # 3. Baseline 3
    print(" >>> 正在生成 Baseline 3 (Multi-ControlNet Pose + Depth)...")
    img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                     controlnet_conditioning_scale=[0.7, 0.5], generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img3.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[2]}.png"))
    clear_gpu_memory()

    # 4. Proposed Method 
    print(" >>> 正在生成 Proposed Method (隔离方案)...")
    masks = process_mask(mask_path)
    apply_attention_mask(pipe_both, token_indices, masks, penalty_weight=20.0)
    img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                     controlnet_conditioning_scale=[0.7, 0.5], generator=generator,
                     num_inference_steps=25, guidance_scale=7.5).images[0]
    img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
    pipe_both.unet.set_attn_processor(AttnProcessor())
    clear_gpu_memory()

    print(f"\n [+] 样本 {sample_id} 的 4 组对照组图像已全部正常生成。")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        clear_gpu_memory()
        raise