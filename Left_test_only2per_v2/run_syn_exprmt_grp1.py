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
TEST_RANGE = [0, 1] 

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

# ===================== 一次性预加载所有 LoRA =====================
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

# ===================== 升级版：高精区域级 LoRA 与交叉注意力双重隔离拦截器 =====================
class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks, char_ids, penalty_weight=20.0):
        super().__init__()
        self.token_indices = token_indices
        self.masks = masks
        self.char_ids = char_ids          # 当前样本对应的两个独立 LoRA 标识名，例如 ["Asuna", "Neferpitou"]
        self.penalty_weight = penalty_weight

    def get_masked_lora_delta(self, layer, adapter_name, states):
        """精准提取特定 LoRA 层的物理网络增量项，并过滤异常基准"""
        if not hasattr(layer, "lora_A") or adapter_name not in layer.lora_A:
            return 0.0
        lora_A = layer.lora_A[adapter_name]
        lora_B = layer.lora_B[adapter_name]
        scaling = layer.scaling[adapter_name]
        
        if hasattr(layer, "lora_dropout") and adapter_name in layer.lora_dropout:
            states = layer.lora_dropout[adapter_name](states)
            
        orig_dtype = states.dtype
        states = states.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(states)).to(orig_dtype) * scaling
        return delta

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        is_self_attn = encoder_hidden_states is None
        
        # 探测当前潜在的特征图空间边长大小
        spatial_size = int(np.sqrt(seq_len)) if int(np.sqrt(seq_len))**2 == seq_len else None
        
        # 1. 动态生成多分辨率图像像素级的空间控制掩码张量
        mask_tensors = []
        if spatial_size in [8, 16, 32, 64] and self.masks:
            for mask in self.masks:
                m_resized = mask.resize((spatial_size, spatial_size), Image.Resampling.NEAREST)
                m_arr = np.array(m_resized) / 255.0
                m_tensor = torch.tensor(m_arr, device=hidden_states.device, dtype=hidden_states.dtype)
                mask_tensors.append(m_tensor.view(1, -1, 1)) # 变换为标准通道广播格式 (1, HW, 1)

        # 2. 拦截并注入计算 Query 向量 (Q 始终代表空间图像像素点)
        if hasattr(attn.to_q, "base_layer") and len(mask_tensors) >= 2 and len(self.char_ids) >= 2:
            q_out = attn.to_q.base_layer(hidden_states)
            q_out += self.get_masked_lora_delta(attn.to_q, self.char_ids[0], hidden_states) * mask_tensors[0]
            q_out += self.get_masked_lora_delta(attn.to_q, self.char_ids[1], hidden_states) * mask_tensors[1]
        else:
            q_out = attn.to_q(hidden_states)

        # 3. 拦截并注入计算 Key 与 Value 向量
        if is_self_attn:
            # 自注意力机制：K 和 V 同样映射自空间像素点
            if hasattr(attn.to_k, "base_layer") and len(mask_tensors) >= 2 and len(self.char_ids) >= 2:
                k_out = attn.to_k.base_layer(hidden_states) + \
                        self.get_masked_lora_delta(attn.to_k, self.char_ids[0], hidden_states) * mask_tensors[0] + \
                        self.get_masked_lora_delta(attn.to_k, self.char_ids[1], hidden_states) * mask_tensors[1]
                v_out = attn.to_v.base_layer(hidden_states) + \
                        self.get_masked_lora_delta(attn.to_v, self.char_ids[0], hidden_states) * mask_tensors[0] + \
                        self.get_masked_lora_delta(attn.to_v, self.char_ids[1], hidden_states) * mask_tensors[1]
            else:
                k_out = attn.to_k(hidden_states)
                v_out = attn.to_v(hidden_states)
        else:
            # 交叉注意力机制：K 和 V 映射自提示词 Token 序列空间
            txt_states = encoder_hidden_states
            txt_len = txt_states.shape[1]
            if hasattr(attn.to_k, "base_layer") and self.token_indices and len(self.char_ids) >= 2:
                k_out = attn.to_k.base_layer(txt_states)
                v_out = attn.to_v.base_layer(txt_states)
                
                # 构建文本序列掩码 (1, 77, 1) 确保文本特征仅触发其对应 LoRA 权重
                t_mask1 = torch.zeros((1, txt_len, 1), device=txt_states.device, dtype=txt_states.dtype)
                t_mask2 = torch.zeros((1, txt_len, 1), device=txt_states.device, dtype=txt_states.dtype)
                for idx in self.token_indices[0]:
                    if idx < txt_len: t_mask1[0, idx, 0] = 1.0
                for idx in self.token_indices[1]:
                    if idx < txt_len: t_mask2[0, idx, 0] = 1.0
                    
                k_out += self.get_masked_lora_delta(attn.to_k, self.char_ids[0], txt_states) * t_mask1
                k_out += self.get_masked_lora_delta(attn.to_k, self.char_ids[1], txt_states) * t_mask2
                v_out += self.get_masked_lora_delta(attn.to_v, self.char_ids[0], txt_states) * t_mask1
                v_out += self.get_masked_lora_delta(attn.to_v, self.char_ids[1], txt_states) * t_mask2
            else:
                k_out = attn.to_k(txt_states)
                v_out = attn.to_v(txt_states)

        # 4. 转换注意力维度矩阵与核心计算
        query = attn.head_to_batch_dim(q_out)
        key = attn.head_to_batch_dim(k_out)
        value = attn.head_to_batch_dim(v_out)

        attn_scores = torch.bmm(query, key.transpose(-1, -2)) * attn.scale

        # 空间文本交叉注意力拦截机制 (原论文算法核心)
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

        # 5. 拦截并注入计算注意力输出线性层 (Out Projection 属于图像像素层级)
        if hasattr(attn.to_out[0], "base_layer") and len(mask_tensors) >= 2 and len(self.char_ids) >= 2:
            out_base = attn.to_out[0].base_layer(hidden_states)
            out_base += self.get_masked_lora_delta(attn.to_out[0], self.char_ids[0], hidden_states) * mask_tensors[0]
            out_base += self.get_masked_lora_delta(attn.to_out[0], self.char_ids[1], hidden_states) * mask_tensors[1]
            hidden_states = out_base
        else:
            hidden_states = attn.to_out[0](hidden_states)

        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def apply_attention_mask(pipe, token_indices, masks, char_ids, penalty_weight=20.0):
    pipe.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks, char_ids, penalty_weight))

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 全量数据集批量测试流 =====================
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
        print(f"\n[+] 激活局部区间测试模式！")
        print(f"    当前计划运行原始数据集索引: {start_idx} 到 {end_idx - 1} (本次共执行 {len(configs_to_run)} 组)")
    else:
        start_idx = 0
        configs_to_run = configs
        print(f"\n[>>>] 当前运行全量测试模式，总规模: {total_scale} 组。")

    char_to_filename = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    print("\n>>> Start preloading full character LoRA weight components...")
    for pipe in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(pipe, char_to_filename)

    print(f"\n>>> Preloading completed. 开始遍历当前选定区间...")
    
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

        # 动态激活基础网络中的全局 LoRA 混色环境 (Baseline 使用)
        for pipe in [pipe_base, pipe_pose, pipe_both]:
            pipe.set_adapters([char1_id, char2_id], adapter_weights=[0.8, 0.8])

        # 每次循环开始前，显式强制将拦截器置空，保证 Baseline 恢复经典状态
        pipe_both.unet.set_attn_processor(AttnProcessor())

        pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
        depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
        mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
        
        token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

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

        # ---- Proposed Method ----
        print("    >>> Generating Proposed Method (Cross-Attention Mask Isolation Scheme)...")
        masks = process_mask(mask_path)
        # 此处升级：传入对应的角色标识符，由拦截器在计算时实时在不同空间切分 LoRA 矩阵
        apply_attention_mask(pipe_both, token_indices, masks, char_ids=[char1_id, char2_id], penalty_weight=20.0)
        img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                         controlnet_conditioning_scale=[0.7, 0.5], generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
        
        # 释放当前循环的拦截器
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

    print(f"\n\n [++] Success! 指定测试区间内的对照组图片已全部生成完毕。")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        clear_gpu_memory()
        raise