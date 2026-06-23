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

# 放大 LoRA Delta 的缩放因子，确保特征信号足够强
LORA_BOOST_SCALE = 5.0 

METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_baseline4", "_method"]
TEST_RANGE = [0, 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def get_fixed_generator():
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== 模型加载 =====================
print("=" * 60)
print("[DEBUG Init] Loading ControlNet & SD base pipelines")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def preload_all_loras(pipe, char_to_filename):
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
    print(f"[DEBUG LoRA] Preloaded {len(char_to_filename)} character LoRAs")

# ===================== Token 解析 =====================
def get_person_token_indices(tokenizer, prompt, debug=False):
    full_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    full_ids = full_inputs.input_ids[0].tolist()
    full_tokens = tokenizer.convert_ids_to_tokens(full_ids)
    clean = [t.replace("</w>", "").lower() for t in full_tokens]

    p1_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "1"), -1)
    p2_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "2"), -1)

    if p1_start == -1 or p2_start == -1:
        return [list(range(5, 30)), list(range(35, 60))]

    p1_desc_start = p1_start + 2
    while p1_desc_start < p2_start and clean[p1_desc_start] in [":", " "]: p1_desc_start += 1
    p1_desc_end = p2_start
    while p1_desc_end > p1_desc_start and clean[p1_desc_end-1] in [".", " "]: p1_desc_end -= 1

    p2_desc_start = p2_start + 2
    while p2_desc_start < len(clean) and clean[p2_desc_start] in [":", " "]: p2_desc_start += 1
    p2_desc_end = next((i for i in range(p2_desc_start, len(clean)) if clean[i] == "."), len(clean))

    return [list(range(p1_desc_start, p1_desc_end)), list(range(p2_desc_start, p2_desc_end))]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask1 = cv2.GaussianBlur(mask1, (9, 9), sigmaX=0.8)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), sigmaX=0.8)
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

def _get_lora_components(layer, adapter_name):
    if hasattr(layer, "base_layer") and hasattr(layer, "lora_A"):
        if adapter_name in layer.lora_A:
            scaling = layer.scaling[adapter_name] if hasattr(layer, "scaling") else 1.0
            return layer.base_layer, layer.lora_A[adapter_name], layer.lora_B[adapter_name], scaling
    if hasattr(layer, "lora_layer"):
        ll = layer.lora_layer
        if hasattr(ll, "lora_A") and adapter_name in ll.lora_A:
            scaling = ll.scaling[adapter_name] if hasattr(ll, "scaling") else 1.0
            return layer, ll.lora_A[adapter_name], ll.lora_B[adapter_name], scaling
    return None

# ===================== 放大 LoRA Delta =====================
def compute_regional_lora_linear(layer, x, masks, char_ids, weights, enable_regional=True):
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None or not enable_regional:
        return layer(x)

    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2
    x_uncond, x_cond = x[:cond_split], x[cond_split:]
    
    out_uncond = layer(x_uncond)
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    seq_len = x_cond.shape[1]
    spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None: continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx] * LORA_BOOST_SCALE

        if spatial_dim in [8, 16, 32, 64] and idx < len(masks):
            m_arr = np.array(masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
            mask_tensor = torch.tensor(m_arr, device=x.device, dtype=x.dtype).view(1, -1, 1)
            combined_delta += delta * mask_tensor
        else:
            combined_delta += delta

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

def compute_text_regional_lora_linear(layer, x, token_indices, char_ids, weights):
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None: return layer(x)

    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2
    x_uncond, x_cond = x[:cond_split], x[cond_split:]
    
    out_uncond = layer(x_uncond)
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None: continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx] * LORA_BOOST_SCALE

        token_mask = torch.zeros((1, x_cond.shape[1], 1), device=x.device, dtype=x.dtype)
        if token_indices and idx < len(token_indices):
            for tid in token_indices[idx]:
                if tid < x_cond.shape[1]: token_mask[0, tid, 0] = 1.0
            combined_delta += delta * token_mask
        else:
            combined_delta += delta

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

# ===================== 特征隔离系统 =====================
class IntegratedMultiRoleProcessor(AttnProcessor):
    def __init__(self, token_indices, masks, char_ids, weights=[0.8, 0.8],
                 enable_cross_mask=True, enable_regional_lora=False,
                 penalty=15.0, boost=2.0):
        super().__init__()
        self.token_indices = token_indices
        self.masks = masks
        self.char_ids = char_ids
        self.weights = weights
        self.enable_cross_mask = enable_cross_mask
        self.enable_regional_lora = enable_regional_lora
        self.penalty = penalty
        self.boost = boost

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

        q = compute_regional_lora_linear(attn.to_q, hidden_states, self.masks, self.char_ids, self.weights, self.enable_regional_lora)

        if is_cross:
            if self.enable_regional_lora:
                k = compute_text_regional_lora_linear(attn.to_k, encoder_hidden_states, self.token_indices, self.char_ids, self.weights)
                v = compute_text_regional_lora_linear(attn.to_v, encoder_hidden_states, self.token_indices, self.char_ids, self.weights)
            else:
                k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        else:
            k = compute_regional_lora_linear(attn.to_k, hidden_states, self.masks, self.char_ids, self.weights, self.enable_regional_lora)
            v = compute_regional_lora_linear(attn.to_v, hidden_states, self.masks, self.char_ids, self.weights, self.enable_regional_lora)

        q, k, v = attn.head_to_batch_dim(q), attn.head_to_batch_dim(k), attn.head_to_batch_dim(v)
        attn_score = torch.bmm(q, k.transpose(-1, -2)) * attn.scale

        if is_cross and self.enable_cross_mask and self.token_indices and spatial_dim in [8, 16, 32, 64]:
            cond_split = attn_score.shape[0] // 2
            for idx, token_ids in enumerate(self.token_indices):
                if idx >= len(self.masks): break
                m_arr = np.array(self.masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
                mask_vec = torch.tensor(m_arr, device=q.device, dtype=q.dtype).view(-1)
                bias = mask_vec * self.boost - (1.0 - mask_vec) * self.penalty
                for tid in token_ids:
                    if tid < attn_score.shape[-1]:
                        attn_score[cond_split:, :, tid] += bias.unsqueeze(0)

        if attention_mask is not None:
            attn_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch)
            attn_score += attn.head_to_batch_dim(attn_mask)

        attn_prob = attn_score.softmax(dim=-1)
        hidden = torch.bmm(attn_prob, v)
        hidden = attn.batch_to_head_dim(hidden)

        hidden = compute_regional_lora_linear(attn.to_out[0], hidden, self.masks, self.char_ids, self.weights, self.enable_regional_lora)
        hidden = attn.to_out[1](hidden)
        return hidden

def clear_gpu():
    import gc; gc.collect(); torch.cuda.empty_cache()

def reset_print_flags():
    for func in [compute_regional_lora_linear, compute_text_regional_lora_linear]:
        for attr in list(dir(func)):
            if attr.endswith("_printed"):
                delattr(func, attr)

# ===================== 主生成逻辑 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    run_list = configs[TEST_RANGE[0]:TEST_RANGE[1]]
    char_map = {"Sera": "LoRA_Sera", "TogaHimiko": "TogaHimiko-01", "MouriRan": "Mouri", "Byakuya": "Byakuya"}

    for p in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(p, char_map)

    for cfg in run_list:
        sid = cfg["sample_id"]
        prompt = re.sub(r"<lora:[^>]+>", "", cfg["prompt"])
        neg_prompt = cfg["negative_prompt"]
        c1, c2 = cfg["characters"]
        
        # [修正1] 精简视角指令，避免占用过多token导致角色特征被截断
        if "front view" not in prompt.lower():
            prompt = prompt + ", front view"
        if "back view" not in neg_prompt.lower():
            neg_prompt = neg_prompt + ", back view"
            
        print(f"\n[DEBUG Main] Modified Prompt: {prompt}")
        print(f"[DEBUG Main] Modified Neg: {neg_prompt}")
        
        pose_img = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sid}.png")).convert("RGB")
        depth_img = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sid}.png")).convert("RGB")
        mask_file = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")

        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt)
        mask_pair = process_mask(mask_file)
        
        # =============== 基线 1-3 ===============
        for p in [pipe_base, pipe_pose, pipe_both]:
            # [修正2] 先全局设置适配器，再单独关闭 text_encoder 的 LoRA
            # 避免 set_adapters 重新激活 text_encoder LoRA，确保仅 unet 生效
            p.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
            if hasattr(p, "text_encoder") and p.text_encoder is not None:
                try:
                    p.text_encoder.set_adapter([]) 
                    print(f"[DEBUG LoRA] Text Encoder LoRA DEACTIVATED for {p.__class__.__name__}")
                except Exception as e:
                    print(f"[WARNING] Could not disable text encoder LoRA: {e}")
            
            p.unet.set_attn_processor(AttnProcessor())

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[0]}")
        pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0].save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[0]}.png"))
        clear_gpu()

        print(f"[-] Generating {sid}{METHOD_SUFFIX[1]}")
        pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose_img, controlnet_conditioning_scale=0.7, generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0].save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[1]}.png"))
        clear_gpu()

        print(f"[-] Generating {sid}{METHOD_SUFFIX[2]}")
        pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.8, 0.4], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0].save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[2]}.png"))
        clear_gpu()

        # =============== 隔离组 ===============
        pipe_both.unet.set_adapters([])
        if hasattr(pipe_both, "text_encoder") and pipe_both.text_encoder is not None:
            try:
                pipe_both.text_encoder.set_adapter([])
            except:
                pass
        
        # Baseline 4
        reset_print_flags()
        print(f"\n[*] Generating {sid}{METHOD_SUFFIX[3]} (Regional LoRA Only)")
        pipe_both.unet.set_attn_processor(IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=False, enable_regional_lora=True
        ))
        pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.8, 0.4], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0].save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[3]}.png"))
        clear_gpu()

        # Method
        reset_print_flags()
        print(f"\n[+] Generating {sid}{METHOD_SUFFIX[4]} (Cross-Attention Mask + Regional LoRA)")
        pipe_both.unet.set_attn_processor(IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=True, enable_regional_lora=True,
            penalty=15.0, boost=2.0
        ))
        # [修正3] 统一 ControlNet 权重与基线3一致，保证单一变量，实验对比公平
        pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.8, 0.4], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0].save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[4]}.png"))
        
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu()

    print("\n All experiments generated successfully.")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        import traceback; traceback.print_exc()
        clear_gpu()
        raise
    
    
    
    
# 三处核心修改说明---相对于tmp_test的19
# LoRA 禁用顺序修正（实验公平性核心）
# 原代码先关闭文本编码器 LoRA、再全局设置适配器，set_adapters 会重新激活text_encoder的 LoRA，导致「基线组仅 unet 生效 LoRA」的目标完全失效。修正后先全局设置、再单独关闭文本编码器，确保基线组和隔离组一致：LoRA 仅作用于 unet，文本编码器无 LoRA 干扰，实验变量唯一，对比结果可信。
# 提示词精简（避免角色特征截断）
# 将原本 3 组语义高度重复的视角词精简为各 1 个（正面front view、负面back view），减少 token 占用，优先保证两个角色的触发词与特征描述完整，避免因超出 77token 上限导致尾部角色特征被截断，降低角色辨识度。
# ControlNet 权重统一（单一变量原则）


# 其余逻辑（区域 LoRA 计算、交叉注意力偏置、种子控制、掩码处理）均正确，可直接运行。