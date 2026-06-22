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

print("=" * 60)
print("[DEBUG Config] Global configuration loaded")
print(f"[DEBUG Config] SYNTHETIC_DATA: {SYNTHETIC_DATA}")
print(f"[DEBUG Config] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[DEBUG Config] LORA_WEIGHTS_DIR: {LORA_WEIGHTS_DIR}")
print(f"[DEBUG Config] IMAGE_SIZE: {IMAGE_SIZE}")
print(f"[DEBUG Config] SEED: {SEED}")
print(f"[DEBUG Config] LORA_BOOST_SCALE: {LORA_BOOST_SCALE}")
print(f"[DEBUG Config] METHOD_SUFFIX: {METHOD_SUFFIX}")
print(f"[DEBUG Config] TEST_RANGE: {TEST_RANGE}")
print(f"[DEBUG Config] Device: {device}")
print("=" * 60)

def get_fixed_generator():
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== 模型加载 =====================
print("=" * 60)
print("[DEBUG Init] Loading ControlNet & SD base pipelines")
print(f"[DEBUG Init] Device: {device}")
print(f"[DEBUG Init] Torch version: {torch.__version__}")
print("[DEBUG Init] Loading ControlNet: OpenPose...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
print("[DEBUG Init] ControlNet OpenPose loaded successfully")
print("[DEBUG Init] Loading ControlNet: Depth...")
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)
print("[DEBUG Init] ControlNet Depth loaded successfully")
print("[DEBUG Init] All ControlNet models loaded")

print("[DEBUG Init] Loading base SD pipeline...")
pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
print("[DEBUG Init] Base SD pipeline loaded successfully")

print("[DEBUG Init] Loading Pose ControlNet pipeline...")
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
print("[DEBUG Init] Pose ControlNet pipeline loaded successfully")

print("[DEBUG Init] Loading Pose+Depth dual ControlNet pipeline...")
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)
print("[DEBUG Init] Dual ControlNet pipeline loaded successfully")
print("[DEBUG Init] All SD pipelines loaded")

print("[DEBUG Init] Setting scheduler to UniPCMultistepScheduler for all pipelines...")
for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("[DEBUG Init] Scheduler set to UniPCMultistepScheduler")
print("=" * 60)

def preload_all_loras(pipe, char_to_filename):
    print(f"\n[DEBUG LoRA] ===== Start preloading {len(char_to_filename)} LoRAs =====")
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        print(f"[DEBUG LoRA] Loading adapter '{char_id}' from: {lora_path}")
        if not os.path.exists(lora_path):
            print(f"[ERROR LoRA] File NOT FOUND: {lora_path}")
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
        print(f"[DEBUG LoRA] Successfully loaded adapter: {char_id}")
    print(f"[DEBUG LoRA] Preloaded {len(char_to_filename)} character LoRAs")
    print(f"[DEBUG LoRA] Current active adapters: {pipe.get_active_adapters()}")
    print(f"[DEBUG LoRA] All available adapters: {pipe.get_list_adapters()}")
    print("[DEBUG LoRA] ===== Preloading finished =====")

# ===================== Token 解析 =====================
def get_person_token_indices(tokenizer, prompt, debug=True):
    print("\n[DEBUG Token] ===== Start parsing person token ranges =====")
    print(f"[DEBUG Token] Raw prompt: {prompt}")
    full_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    full_ids = full_inputs.input_ids[0].tolist()
    full_tokens = tokenizer.convert_ids_to_tokens(full_ids)
    clean = [t.replace("</w>", "").lower() for t in full_tokens]

    print(f"[DEBUG Token] Total token sequence length: {len(full_tokens)}")
    if debug:
        print("\n[DEBUG Token] Full token list (index: token):")
        for i in range(min(75, len(full_tokens))):
            if full_tokens[i] == "<|endoftext|>":
                break
            print(f"  {i:2d}: {full_tokens[i]}")

    p1_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "1"), -1)
    p2_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "2"), -1)

    print(f"\n[DEBUG Token] Detected marker positions:")
    print(f"  person1 marker starts at index: {p1_start}")
    print(f"  person2 marker starts at index: {p2_start}")

    if p1_start == -1 or p2_start == -1:
        print("[WARNING Token] Cannot find person markers, falling back to hardcoded range")
        fallback = [list(range(5, 30)), list(range(35, 60))]
        print(f"[DEBUG Token] Fallback ranges: person1={fallback[0][0]}~{fallback[0][-1]}, person2={fallback[1][0]}~{fallback[1][-1]}")
        return fallback

    p1_desc_start = p1_start + 2
    while p1_desc_start < p2_start and clean[p1_desc_start] in [":", " "]: p1_desc_start += 1
    p1_desc_end = p2_start
    while p1_desc_end > p1_desc_start and clean[p1_desc_end-1] in [".", " "]: p1_desc_end -= 1

    p2_desc_start = p2_start + 2
    while p2_desc_start < len(clean) and clean[p2_desc_start] in [":", " "]: p2_desc_start += 1
    p2_desc_end = next((i for i in range(p2_desc_start, len(clean)) if clean[i] == "."), len(clean))

    p1_range = list(range(p1_desc_start, p1_desc_end))
    p2_range = list(range(p2_desc_start, p2_desc_end))

    print(f"\n[DEBUG Token] Final aligned ranges:")
    print(f"  person1 tokens: {len(p1_range)} -> indices {p1_range[0]} ~ {p1_range[-1]}")
    print(f"  person2 tokens: {len(p2_range)} -> indices {p2_range[0]} ~ {p2_range[-1]}")

    if debug:
        print("\n[DEBUG Token] Person1 actual tokens:")
        print(f"  {[full_tokens[i] for i in p1_range]}")
        print("[DEBUG Token] Person2 actual tokens:")
        print(f"  {[full_tokens[i] for i in p2_range]}")

    print("[DEBUG Token] ===== Token parsing finished =====")
    return [p1_range, p2_range]

def process_mask(mask_path):
    print(f"\n[DEBUG Mask] ===== Start processing mask: {mask_path} =====")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR Mask] Failed to read mask image!")
        raise FileNotFoundError(f"Mask not found or invalid: {mask_path}")
    print(f"[DEBUG Mask] Raw mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"[DEBUG Mask] Unique pixel values in raw mask: {np.unique(mask)}")

    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    print(f"[DEBUG Mask] mask1 (value 128) non-zero pixels: {np.count_nonzero(mask1)}")
    print(f"[DEBUG Mask] mask2 (value 255) non-zero pixels: {np.count_nonzero(mask2)}")

    if np.count_nonzero(mask1) == 0:
        print("[WARNING Mask] mask1 (value 128) is EMPTY!")
    if np.count_nonzero(mask2) == 0:
        print("[WARNING Mask] mask2 (value 255) is EMPTY!")

    kernel = np.ones((5, 5), np.uint8)
    print("[DEBUG Mask] Applying dilation (kernel 5x5, iterations=1)...")
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    print("[DEBUG Mask] Applied dilation")

    print("[DEBUG Mask] Applying Gaussian blur (kernel 9x9, sigmaX=0.8)...")
    mask1 = cv2.GaussianBlur(mask1, (9, 9), sigmaX=0.8)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), sigmaX=0.8)
    print("[DEBUG Mask] Applied Gaussian blur")

    print(f"[DEBUG Mask] Final mask1 value range: [{mask1.min():.2f}, {mask1.max():.2f}]")
    print(f"[DEBUG Mask] Final mask2 value range: [{mask2.min():.2f}, {mask2.max():.2f}]")
    print("[DEBUG Mask] ===== Mask processing finished =====")
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
    if not hasattr(compute_regional_lora_linear, "_entry_printed"):
        compute_regional_lora_linear._entry_printed = True
        print("\n[DEBUG Reg LoRA] ===== Enter compute_regional_lora_linear =====")
        print(f"[DEBUG Reg LoRA] enable_regional: {enable_regional}")
        print(f"[DEBUG Reg LoRA] Input x shape: {x.shape}")
        print(f"[DEBUG Reg LoRA] Character IDs: {char_ids}")
        print(f"[DEBUG Reg LoRA] Weights: {weights}")

    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None or not enable_regional:
        if not hasattr(compute_regional_lora_linear, "_skip_printed"):
            compute_regional_lora_linear._skip_printed = True
            print("[DEBUG Reg LoRA] LoRA components not found or regional disabled, returning raw layer output")
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

    if spatial_dim in [8, 16, 32, 64] and not hasattr(compute_regional_lora_linear, f"_dim_{spatial_dim}_printed"):
        setattr(compute_regional_lora_linear, f"_dim_{spatial_dim}_printed", True)
        print(f"[DEBUG Reg LoRA] Detected spatial resolution: {spatial_dim}x{spatial_dim}")
        print(f"[DEBUG Reg LoRA] base_out_cond mean (abs): {base_out_cond.abs().mean():.6f}")

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            print(f"[WARNING Reg LoRA] Cannot get components for {cid}")
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx] * LORA_BOOST_SCALE

        if spatial_dim in [8, 16, 32, 64] and idx < len(masks):
            m_arr = np.array(masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
            mask_tensor = torch.tensor(m_arr, device=x.device, dtype=x.dtype).view(1, -1, 1)
            combined_delta += delta * mask_tensor
            
            if not hasattr(compute_regional_lora_linear, f"_char_{idx}_delta_printed"):
                setattr(compute_regional_lora_linear, f"_char_{idx}_delta_printed", True)
                print(f"[DEBUG Reg LoRA] Char {idx} ({cid}) - delta mean (abs): {delta.abs().mean():.6f}, mask mean: {mask_tensor.mean():.6f}")
        else:
            combined_delta += delta

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

def compute_text_regional_lora_linear(layer, x, token_indices, char_ids, weights):
    if not hasattr(compute_text_regional_lora_linear, "_entry_printed"):
        compute_text_regional_lora_linear._entry_printed = True
        print("\n[DEBUG Text LoRA] ===== Enter compute_text_regional_lora_linear =====")
        print(f"[DEBUG Text LoRA] Input x shape: {x.shape}")
        print(f"[DEBUG Text LoRA] Character IDs: {char_ids}")
        print(f"[DEBUG Text LoRA] Weights: {weights}")

    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None:
        print("[WARNING Text LoRA] LoRA components not found, returning raw layer output")
        return layer(x)

    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2
    x_uncond, x_cond = x[:cond_split], x[cond_split:]
    
    out_uncond = layer(x_uncond)
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    if not hasattr(compute_text_regional_lora_linear, "_base_printed"):
        compute_text_regional_lora_linear._base_printed = True
        print(f"[DEBUG Text LoRA] base_out_cond mean (abs): {base_out_cond.abs().mean():.6f}")

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            print(f"[WARNING Text LoRA] Cannot get components for {cid}")
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx] * LORA_BOOST_SCALE

        token_mask = torch.zeros((1, x_cond.shape[1], 1), device=x.device, dtype=x.dtype)
        if token_indices and idx < len(token_indices):
            for tid in token_indices[idx]:
                if tid < x_cond.shape[1]: token_mask[0, tid, 0] = 1.0
            combined_delta += delta * token_mask
            
            if not hasattr(compute_text_regional_lora_linear, f"_char_{idx}_mask_printed"):
                setattr(compute_text_regional_lora_linear, f"_char_{idx}_mask_printed", True)
                print(f"[DEBUG Text LoRA] Char {idx} ({cid}) - token mask sum: {token_mask.sum().item()}, delta mean (abs): {delta.abs().mean():.6f}")
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

        print(f"\n[DEBUG Processor] IntegratedMultiRoleProcessor initialized")
        print(f"[DEBUG Processor] enable_cross_mask: {enable_cross_mask}")
        print(f"[DEBUG Processor] enable_regional_lora: {enable_regional_lora}")
        print(f"[DEBUG Processor] penalty: {penalty}, boost: {boost}")
        print(f"[DEBUG Processor] char_ids: {char_ids}")
        print(f"[DEBUG Processor] token_indices lengths: {[len(t) for t in token_indices]}")
        print(f"[DEBUG Processor] masks count: {len(masks)}")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

        if not is_cross and not hasattr(self, "_self_first_printed"):
            self._self_first_printed = True
            print(f"\n[DEBUG Processor] First self-attention forward")
            print(f"[DEBUG Processor] hidden_states shape: {hidden_states.shape}")
            print(f"[DEBUG Processor] spatial_dim: {spatial_dim}")

        if is_cross and not hasattr(self, "_cross_first_printed"):
            self._cross_first_printed = True
            print(f"\n[DEBUG Processor] First cross-attention forward")
            print(f"[DEBUG Processor] hidden_states shape: {hidden_states.shape}")
            print(f"[DEBUG Processor] encoder_hidden_states shape: {encoder_hidden_states.shape}")
            print(f"[DEBUG Processor] spatial_dim: {spatial_dim}")

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
            if not hasattr(self, "_bias_start_printed"):
                self._bias_start_printed = True
                print(f"\n[DEBUG Processor] Applying cross-attention bias at resolution {spatial_dim}")
                print(f"[DEBUG Processor] Raw attn_score range: [{attn_score.min():.2f}, {attn_score.max():.2f}], mean: {attn_score.mean():.2f}")

            cond_split = attn_score.shape[0] // 2
            for idx, token_ids in enumerate(self.token_indices):
                if idx >= len(self.masks): break
                m_arr = np.array(self.masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
                mask_vec = torch.tensor(m_arr, device=q.device, dtype=q.dtype).view(-1)
                bias = mask_vec * self.boost - (1.0 - mask_vec) * self.penalty
                
                if not hasattr(self, f"_bias_stats_{idx}"):
                    setattr(self, f"_bias_stats_{idx}", True)
                    print(f"[DEBUG Processor] Char {idx} bias: min={bias.min():.2f}, max={bias.max():.2f}, mean={bias.mean():.2f}")

                for tid in token_ids:
                    if tid < attn_score.shape[-1]:
                        attn_score[cond_split:, :, tid] += bias.unsqueeze(0)

            if not hasattr(self, "_after_bias_printed"):
                self._after_bias_printed = True
                print(f"[DEBUG Processor] After bias: attn_score range: [{attn_score.min():.2f}, {attn_score.max():.2f}], mean: {attn_score.mean():.2f}")

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
    print("\n[DEBUG GPU] ===== Clearing GPU cache =====")
    import gc; gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1024**3
        reserv_gb = torch.cuda.memory_reserved() / 1024**3
        print(f"[DEBUG GPU] Memory allocated: {alloc_gb:.2f} GB, reserved: {reserv_gb:.2f} GB")
    print("[DEBUG GPU] ===== GPU cache cleared =====")

def reset_print_flags():
    print("\n[DEBUG Utils] ===== Resetting all debug print flags =====")
    count = 0
    for func in [compute_regional_lora_linear, compute_text_regional_lora_linear]:
        for attr in list(dir(func)):
            if attr.endswith("_printed"):
                delattr(func, attr)
                count += 1
    print(f"[DEBUG Utils] Reset {count} print flags")
    print("[DEBUG Utils] ===== Print flags reset complete =====")

# ===================== 主生成逻辑 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    print(f"\n[DEBUG Main] Loading config from: {config_path}")
    if not os.path.exists(config_path):
        print(f"[ERROR Main] Config file NOT FOUND: {config_path}")
        raise FileNotFoundError(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    print(f"[DEBUG Main] Total samples in config: {len(configs)}")
    start_idx, end_idx = TEST_RANGE
    run_list = configs[start_idx:end_idx]
    print(f"[DEBUG Main] Test range: [{start_idx}:{end_idx}], running {len(run_list)} samples")
    print(f"[DEBUG Main] First sample ID: {run_list[0]['sample_id']}")

    char_map = {"Sera": "LoRA_Sera", "TogaHimiko": "TogaHimiko-01", "MouriRan": "Mouri", "Byakuya": "Byakuya"}
    print(f"[DEBUG Main] Character map: {char_map}")

    for p in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(p, char_map)

    print("\n" + "="*60)
    print("[DEBUG Main] Global LoRA preload check:")
    print(f"  pipe_base active adapters: {pipe_base.get_active_adapters()}")
    print(f"  pipe_pose active adapters: {pipe_pose.get_active_adapters()}")
    print(f"  pipe_both active adapters: {pipe_both.get_active_adapters()}")
    print("="*60)

    for cfg in run_list:
        sid = cfg["sample_id"]
        print(f"\n{'='*60}")
        print(f"[DEBUG Main] Processing sample: {sid}")
        print(f"{'='*60}")

        prompt = re.sub(r"<lora:[^>]+>", "", cfg["prompt"])
        prompt = re.sub(r",? no [A-Za-z]+ features", "", prompt)
        neg_prompt = cfg["negative_prompt"]
        c1, c2 = cfg["characters"]
        print(f"[DEBUG Main] Characters: {c1} + {c2}")
        print(f"[DEBUG Main] Negative prompt: {neg_prompt}")
        
        # 强制注入正面视角指令，解决“背对画面”问题
        if "facing viewer" not in prompt.lower() and "front view" not in prompt.lower():
            prompt = prompt + ", facing viewer, front view, looking at camera"
        if "from behind" not in neg_prompt.lower() and "back view" not in neg_prompt.lower():
            neg_prompt = neg_prompt + ", from behind, back view, facing away"
            
        print(f"[DEBUG Main] Modified Prompt: {prompt}")
        print(f"[DEBUG Main] Modified Neg: {neg_prompt}")
        
        pose_path = os.path.join(SYNTHETIC_DATA, "poses", f"{sid}.png")
        depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sid}.png")
        mask_file = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")
        print(f"[DEBUG Main] Pose file exists: {os.path.exists(pose_path)}")
        print(f"[DEBUG Main] Depth file exists: {os.path.exists(depth_path)}")
        print(f"[DEBUG Main] Mask file exists: {os.path.exists(mask_file)}")

        pose_img = Image.open(pose_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("RGB")
        print(f"[DEBUG Main] Pose image size: {pose_img.size}")
        print(f"[DEBUG Main] Depth image size: {depth_img.size}")

        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt, debug=True)
        mask_pair = process_mask(mask_file)
        
        # =============== 基线 1-3 ===============
        print("\n--- Switching to GLOBAL LoRA mode (baselines 1-3) ---")
        for p in [pipe_base, pipe_pose, pipe_both]:
            # 【   1】先彻底切断 Text Encoder 的 LoRA，防止文本特征预先混合
            if hasattr(p, "text_encoder") and p.text_encoder is not None:
                try:
                    p.text_encoder.set_adapter([]) 
                    print(f"[DEBUG LoRA] Text Encoder LoRA DEACTIVATED for {p.__class__.__name__}")
                except Exception as e:
                    print(f"[WARNING] Could not disable text encoder LoRA: {e}")
            
            # 【   】在 Pipeline 级别 (p) 设置 adapters 和 weights，而不是 p.unet
            p.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
            p.unet.set_attn_processor(AttnProcessor())

        print(f"[DEBUG Main] Baseline LoRA status - active adapters: {pipe_both.get_active_adapters()}")
        print(f"[DEBUG Main] text_encoder active adapters: {pipe_both.text_encoder.active_adapters}")

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[0]}")
        img1 = pipe_base(
            prompt=prompt, negative_prompt=neg_prompt,
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save1 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[0]}.png")
        img1.save(save1)
        print(f"[DEBUG Main] Saved to: {save1}")
        clear_gpu()

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[1]}")
        img2 = pipe_pose(
            prompt=prompt, negative_prompt=neg_prompt,
            image=pose_img, controlnet_conditioning_scale=0.7,
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save2 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[1]}.png")
        img2.save(save2)
        print(f"[DEBUG Main] Saved to: {save2}")
        clear_gpu()

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[2]}")
        img3 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt,
            image=[pose_img, depth_img],
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save3 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[2]}.png")
        img3.save(save3)
        print(f"[DEBUG Main] Saved to: {save3}")
        clear_gpu()
        
        # =============== 隔离组 ===============
        print("\n--- Switching to REGIONAL LoRA mode ---")
        pipe_both.unet.set_adapters([])
        if hasattr(pipe_both, "text_encoder") and pipe_both.text_encoder is not None:
            try:
                pipe_both.text_encoder.set_adapter([])
            except:
                pass
                
        print(f"[DEBUG Main] pipeline active adapters: {pipe_both.get_active_adapters()}")
        print(f"[DEBUG Main] text_encoder active adapters: {pipe_both.text_encoder.active_adapters}")

        # Baseline 4
        reset_print_flags()
        print(f"\n[*] Generating {sid}{METHOD_SUFFIX[3]} (Regional LoRA Only)")
        processor4 = IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=False, enable_regional_lora=True
        )
        pipe_both.unet.set_attn_processor(processor4)
        img4 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt,
            image=[pose_img, depth_img],
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save4 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[3]}.png")
        img4.save(save4)
        print(f"[DEBUG Main] Saved to: {save4}")
        clear_gpu()

        # Method
        reset_print_flags()
        print(f"\n[+] Generating {sid}{METHOD_SUFFIX[4]} (Cross-Attention Mask + Regional LoRA)")
        processor_method = IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=True, enable_regional_lora=True,
            penalty=15.0, boost=2.0
        )
        
        pipe_both.unet.set_attn_processor(processor_method)
        img5 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt,
            image=[pose_img, depth_img],
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save5 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[4]}.png")
        img5.save(save5)
        print(f"[DEBUG Main] Saved to: {save5}")
        
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu()

    print("\n" + "="*60)
    print(" All experiments generated successfully.")
    print("="*60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[DEBUG Entry] Script execution started")
    print("="*60)
    try:
        run_synthetic()
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu()
        raise