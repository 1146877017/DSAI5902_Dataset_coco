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

# 3组基线 + 1组消融 + 1组主方法
METHOD_SUFFIX = [
    "_baseline1",          # SD + 全局双LoRA
    "_baseline2",          # SD + Pose + 全局双LoRA
    "_baseline3",          # SD + Pose + Depth + 全局双LoRA
    "_baseline4",          # SD + Pose + Depth + 仅区域LoRA 
    "_method"              # SD + Pose + Depth + 区域LoRA + 交叉注意力掩码 
]
TEST_RANGE = [0, 1]  

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# 核心修复 1：每次生成都重新实例化生成器，确保初始隐空间噪声 100% 对齐
def get_fixed_generator():
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== 模型加载 =====================
print("=" * 60)
print(" [+] Loading ControlNet & SD base pipelines")
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
    print(f"Preloaded {len(char_to_filename)} character LoRAs")

# 核心修复 2：增强型 BPE 文本拦截，确保特定角色的特征词被精准框定
def get_person_token_indices(tokenizer, prompt, debug=False):
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower().strip(":").strip() for t in tokens]

    p1_start = p1_end = p2_start = p2_end = -1

    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person1" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "1"):
            p1_start = i
            for j in range(i + 1, len(clean_tokens)):
                if clean_tokens[j] == "person2" or clean_tokens[j] == "<|endoftext|>":
                    p1_end = j
                    break
            break

    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person2" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "2"):
            p2_start = i
            for j in range(i + 1, len(clean_tokens)):
                if clean_tokens[j] == "<|endoftext|>" or any(w in clean_tokens[j] for w in ["scene", "background", "park", "room", "street"]):
                    p2_end = j
                    break
            break

    if p1_start != -1 and p1_end != -1 and p2_start != -1 and p2_end != -1 and p2_start >= p1_end:
        return [list(range(p1_start, p1_end)), list(range(p2_start, p2_end))]
    
    print("⚠️ Token parse failed, using fallback range")
    return [list(range(5, 20)), list(range(25, 45))]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate((mask == 128).astype(np.uint8) * 255, kernel, iterations=1)
    mask2 = cv2.dilate((mask == 255).astype(np.uint8) * 255, kernel, iterations=1)
    mask1 = cv2.GaussianBlur(mask1, (9, 9), sigmaX=1.5)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), sigmaX=1.5)
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

def compute_regional_lora_linear(layer, x, masks, char_ids, weights, enable_regional=True):
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None or not enable_regional:
        return layer(x)

    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2

    x_uncond = x[:cond_split]
    x_cond = x[cond_split:]
    out_uncond = layer(x_uncond)
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    seq_len = x_cond.shape[1]
    spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx]

        if spatial_dim in [8, 16, 32, 64] and idx < len(masks):
            m_arr = np.array(masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
            mask_tensor = torch.tensor(m_arr, device=x.device, dtype=x.dtype).view(1, -1, 1)
            combined_delta += delta * mask_tensor
        else:
            combined_delta += delta * 0.5

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

def compute_text_regional_lora_linear(layer, x, token_indices, char_ids, weights):
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None:
        return layer(x)

    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2

    x_uncond = x[:cond_split]
    x_cond = x[cond_split:]
    out_uncond = layer(x_uncond)
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx]

        token_mask = torch.zeros((1, x_cond.shape[1], 1), device=x.device, dtype=x.dtype)
        if token_indices and idx < len(token_indices):
            for tid in token_indices[idx]:
                if tid < x_cond.shape[1]:
                    token_mask[0, tid, 0] = 1.0
            combined_delta += delta * token_mask
        else:
            combined_delta += delta * 0.5

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

# ===================== 核心优化 3：双管齐下的特征隔离系统 =====================
class IntegratedMultiRoleProcessor(AttnProcessor):
    def __init__(
        self, token_indices, masks, char_ids, weights=[0.8, 0.8],
        enable_cross_mask=True, enable_regional_lora=False,
        penalty=25.0, boost=6.0
    ):
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
                k = attn.to_k(encoder_hidden_states)
                v = attn.to_v(encoder_hidden_states)
        else:
            k = compute_regional_lora_linear(attn.to_k, hidden_states, self.masks, self.char_ids, self.weights, self.enable_regional_lora)
            v = compute_regional_lora_linear(attn.to_v, hidden_states, self.masks, self.char_ids, self.weights, self.enable_regional_lora)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

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
            attn_mask = attn.head_to_batch_dim(attn_mask)
            attn_score += attn_mask

        attn_prob = attn_score.softmax(dim=-1)
        hidden = torch.bmm(attn_prob, v)
        hidden = attn.batch_to_head_dim(hidden)

        hidden = compute_regional_lora_linear(attn.to_out[0], hidden, self.masks, self.char_ids, self.weights, self.enable_regional_lora)
        hidden = attn.to_out[1](hidden)
        return hidden

def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 主生成逻辑 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    start_idx, end_idx = TEST_RANGE
    run_list = configs[start_idx:end_idx]

    char_map = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    for p in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(p, char_map)

    for cfg in run_list:
        sid = cfg["sample_id"]
        prompt = re.sub(r"<lora:[^>]+>", "", cfg["prompt"])
        prompt = re.sub(r",? no [A-Za-z]+ features", "", prompt)
        neg_prompt = cfg["negative_prompt"]
        c1, c2 = cfg["characters"]
        pose_img = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sid}.png")).convert("RGB")
        depth_img = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sid}.png")).convert("RGB")
        mask_file = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")

        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt, debug=True)
        mask_pair = process_mask(mask_file)

        # =============== 阶段A：基线测试（应用全局污染 LoRA） ===============
        for p in [pipe_base, pipe_pose, pipe_both]:
            p.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
            p.unet.set_attn_processor(AttnProcessor())

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[0]}")
        img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0]
        img1.save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[0]}.png"))
        clear_gpu()

        print(f"[-] Generating {sid}{METHOD_SUFFIX[1]}")
        img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose_img, controlnet_conditioning_scale=0.7, generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0]
        img2.save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[1]}.png"))
        clear_gpu()

        print(f"[-] Generating {sid}{METHOD_SUFFIX[2]}")
        img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.7, 0.5], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0]
        img3.save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[2]}.png"))
        clear_gpu()

        # =============== 阶段B：学术隔离机制组（卸载全局 LoRA，交由底层拦截器分发） ===============
        pipe_both.set_adapters([]) 

        # Baseline 4 (Ablation)：仅区域 LoRA 控制，关闭交叉注意力惩罚
        print(f"[*] Generating {sid}{METHOD_SUFFIX[3]} (Ablation: Regional LoRA Only)")
        pipe_both.unet.set_attn_processor(
            IntegratedMultiRoleProcessor(
                token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
                enable_cross_mask=False, enable_regional_lora=True
            )
        )
        img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.7, 0.5], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[3]}.png"))
        clear_gpu()

        # Proposed Method：融合交叉注意力掩码与区域 LoRA 控制，实现完美物理防泄漏
        print(f"[+] Generating {sid}{METHOD_SUFFIX[4]} (Proposed: Cross-Attention Mask + Regional LoRA)")
        pipe_both.unet.set_attn_processor(
            IntegratedMultiRoleProcessor(
                token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
                enable_cross_mask=True, enable_regional_lora=True, penalty=20.0, boost=1.5
            )
        )
        img5 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose_img, depth_img], controlnet_conditioning_scale=[0.7, 0.5], generator=get_fixed_generator(), num_inference_steps=25, guidance_scale=7.5).images[0]
        img5.save(os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[4]}.png"))
        
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu()

    print("\n[✔] All experiments generated successfully with high-fidelity de-bleeding setup.")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        clear_gpu()
        raise