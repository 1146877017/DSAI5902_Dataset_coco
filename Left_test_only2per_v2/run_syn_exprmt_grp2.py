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

# ===================== 测试范围控制 =====================
TEST_RANGE = [1, 2] 

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
generator = torch.Generator(device).manual_seed(SEED)

# ===================== 加载模型 =====================
print("=" * 60)
print(" [+] Parallel loading OpenPose official color coding standard and computing flow...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# ===================== 预加载所有 LoRA =====================
def preload_all_loras(pipe, char_to_filename):
    """在循环外部将所有可能用到的 4 个 LoRA 权重全部常驻显存"""
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Missing LoRA weights: '{lora_path}'")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
    print(f" [+] Successfully preloaded {len(char_to_filename)} character LoRA matrices for the current Pipeline.")

def get_person_token_indices(tokenizer, prompt):
    """精准切分 person1 描述区间与 person2 描述区间"""
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
            person1_token_ids = list(range(p1_start, p2_start))
            person2_token_ids = list(range(p2_start, scene_start))
            return [person1_token_ids, person2_token_ids]
    except Exception as e:
        print(f"[-] Exception in dynamic token boundary parsing, triggering safety backup mechanism: {e}")
        
    return [list(range(5, 15)), list(range(16, 26))]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

# ===================== 强隔离型双域拦截处理器 =====================
class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks, char_ids, penalty_weight=100.0):
        super().__init__()
        self.token_indices = token_indices
        self.masks = masks
        self.char_ids = char_ids  # 当前样本激活的两个角色 ID 列表
        self.penalty_weight = penalty_weight

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        is_self_attn = encoder_hidden_states is None
        spatial_size = int(np.sqrt(seq_len)) if int(np.sqrt(seq_len))**2 == seq_len else None

        # ----------------- 解耦逻辑：手动进行区域及 Token 级 LoRA 前向计算 -----------------
        def forward_regional_lora(linear_layer, x, is_spatial=True, s_size=spatial_size):
            # 基础网络计算（此时全局 adapter 已关，得到干净的 base 特征）
            base_out = linear_layer(x)
            if self.char_ids is None or not hasattr(linear_layer, "lora_A"):
                return base_out
            
            lora_out = torch.zeros_like(base_out)
            for i, char_id in enumerate(self.char_ids):
                if char_id in linear_layer.lora_A:
                    lora_A = linear_layer.lora_A[char_id]
                    lora_B = linear_layer.lora_B[char_id]
                    scaling = linear_layer.scaling[char_id]
                    dropout = linear_layer.lora_dropout[char_id] if hasattr(linear_layer, "lora_dropout") else lambda y: y
                    
                    # 计算该角色的原始 LoRA 增量
                    delta = lora_B(lora_A(dropout(x))) * scaling
                    
                    if is_spatial and self.masks is not None and i < len(self.masks) and s_size in [8, 16, 32, 64]:
                        # 【空间域强隔离】Latent 特征图按 Mask 赋予 LoRA 增量
                        m_arr = np.array(self.masks[i].resize((s_size, s_size), Image.Resampling.NEAREST)) / 255.0
                        mask_tensor = torch.tensor(m_arr, device=x.device, dtype=x.dtype).view(1, -1, 1)
                        lora_out += delta * mask_tensor
                    elif not is_spatial and self.token_indices is not None and i < len(self.token_indices):
                        # 【文本域强隔离】Prompt Token 序列按位置绑定激活 LoRA
                        token_mask = torch.zeros((x.shape[1], 1), device=x.device, dtype=x.dtype)
                        for t_idx in self.token_indices[i]:
                            if t_idx < token_mask.shape[0]:
                                token_mask[t_idx, 0] = 1.0
                        lora_out += delta * token_mask.unsqueeze(0)
                    else:
                        # 非指定层或非核心分辨率，非空间域则默认全局应用（用于非交叉注意力的文本编码补充）
                        if not is_spatial:
                            lora_out += delta
                            
            return base_out + lora_out

        # 分别对 Q, K, V 应用精细化的区域 LoRA 矩阵运算
        if is_self_attn:
            q_out = forward_regional_lora(attn.to_q, hidden_states, is_spatial=True)
            k_out = forward_regional_lora(attn.to_k, hidden_states, is_spatial=True)
            v_out = forward_regional_lora(attn.to_v, hidden_states, is_spatial=True)
        else:
            q_out = forward_regional_lora(attn.to_q, hidden_states, is_spatial=True)
            k_out = forward_regional_lora(attn.to_k, encoder_hidden_states, is_spatial=False)
            v_out = forward_regional_lora(attn.to_v, encoder_hidden_states, is_spatial=False)

        query = attn.head_to_batch_dim(q_out)
        key = attn.head_to_batch_dim(k_out)
        value = attn.head_to_batch_dim(v_out)

        attn_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale

        # 空间文本交叉注意力布局惩罚机制
        if not is_self_attn and self.token_indices and spatial_size in [8, 16, 32, 64]:
            chunks = attn_scores.shape[0] // 2
            for i, token_ids in enumerate(self.token_indices):
                if i >= len(self.masks): break
                m_arr = np.array(self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.NEAREST)) / 255.0
                mask_tensor = torch.tensor(m_arr, device=query.device, dtype=query.dtype)
                penalty_vector = (1.0 - mask_tensor).view(-1) * (-self.penalty_weight)
                for token_idx in token_ids:
                    if token_idx >= attn_scores.shape[-1]: continue
                    attn_scores[chunks:, :, token_idx] += penalty_vector

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attn.head_to_batch_dim(attention_mask)
            attn_scores = attn_scores + attention_mask

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 最后的注意力输出线性层同样进行空间局域化 LoRA 映射
        hidden_states = forward_regional_lora(attn.to_out[0], hidden_states, is_spatial=True)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def apply_attention_mask(pipe, token_indices, masks, char_ids, penalty_weight=100.0):
    pipe.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks, char_ids, penalty_weight))

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 批量测试 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    total_scale = len(configs)
    if total_scale == 0:
        print("[-] No samples found in the config file.")
        return

    if TEST_RANGE is not None:
        start_idx, end_idx = TEST_RANGE
        configs_to_run = configs[start_idx:end_idx]
        print(f"\n[+] Activating local interval test mode")
        print(f"    Currently planned to run original dataset indices: {start_idx} to {end_idx - 1} (Total of {len(configs_to_run)} groups executed this time)")
    else:
        start_idx = 0
        configs_to_run = configs
        print(f"\n[>>>] Currently running full test mode, total scale: {total_scale} groups.")

    char_to_filename = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    print("\n>>> Start preloading full character LoRA weight components...")
    for pipe in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(pipe, char_to_filename)

    print(f"\n>>> Preloading completed. Starting to iterate through the currently selected interval...")
    
    for idx_offset, cfg in enumerate(configs_to_run):
        actual_index = start_idx + idx_offset
        
        sample_id = cfg["sample_id"]
        prompt = cfg["prompt"]
        neg_prompt = cfg["negative_prompt"]

        char1_id = cfg["characters"][0]
        char2_id = cfg["characters"][1]

        prompt = re.sub(r"<lora:[^>]+>", "", prompt)
        print(f"\n" + "="*50)
        print(f" [{actual_index + 1}/{total_scale}] Processing multi-role academic intersection pipeline -> Sample ID: {sample_id}")
        print(f" [+] Current character pair: {char1_id} (Person1) vs {char2_id} (Person2)")

        pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
        depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
        mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
        
        token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

        # ------------------ Baseline 组测试 ------------------
        # 显式激活全局多 LoRA 混色环境
        for pipe in [pipe_base, pipe_pose, pipe_both]:
            pipe.set_adapters([char1_id, char2_id], adapter_weights=[1.0, 1.0])
            pipe.unet.set_attn_processor(AttnProcessor())  # 基线清空拦截器

        # ---- Baseline 1 ----
        print("    >>> Generating Baseline 1 (Base Pipeline)...")
        img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[0]}.png"))
        clear_gpu_memory()

        # ---- Baseline 2 ----
        print("    >>> Generating Baseline 2 (Single ControlNet Pose)...")
        img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose,
                         controlnet_conditioning_scale=0.7, generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[1]}.png"))
        clear_gpu_memory()

        # ---- Baseline 3 ----
        print("    >>> Generating Baseline 3 (Multi-ControlNet Pose + Depth)...")
        img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                         controlnet_conditioning_scale=[0.7, 0.5], generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img3.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[2]}.png"))
        clear_gpu_memory()

        # ------------------ Proposed Method 组测试 ------------------
        print("    >>> Generating Proposed Method (Dual-Domain LoRA Isolation Scheme)...")
        masks = process_mask(mask_path)
        
        # 1. 净化基座：关闭全局加载的 active adapters，使其不产生全局合并污染
        pipe_both.set_adapters([])
        
        # 2. 注入解耦拦截器：传递当前对拷的 char_ids，在运算流中内部进行局部动态矩阵激活
        apply_attention_mask(pipe_both, token_indices, masks, char_ids=[char1_id, char2_id], penalty_weight=100.0)
        
        img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                         controlnet_conditioning_scale=[0.7, 0.5], generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
        
        # 释放当前循环的拦截器
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

    print(f"\n\n [++] Success! All control group images within the designated test interval have been generated.")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        clear_gpu_memory()
        raise