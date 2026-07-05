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
print("[INIT] Loading global configuration...")
SYNTHETIC_DATA = "synthetic_test_dataset"
OUTPUT_DIR = "synthetic_results"
LORA_WEIGHTS_DIR = "./lora_weights"
IMAGE_SIZE = 512
SEED = 42

# 放大 LoRA Delta 的缩放因子，确保特征信号足够强
LORA_BOOST_SCALE = 5.0 

METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_baseline4", "_method"]
TEST_RANGE = [55, 60]

print(f"[INIT] SYNTHETIC_DATA: {SYNTHETIC_DATA}")
print(f"[INIT] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[INIT] LORA_WEIGHTS_DIR: {LORA_WEIGHTS_DIR}")
print(f"[INIT] IMAGE_SIZE: {IMAGE_SIZE}, SEED: {SEED}")
print(f"[INIT] LORA_BOOST_SCALE: {LORA_BOOST_SCALE}")
print(f"[INIT] METHOD_SUFFIX: {METHOD_SUFFIX}")
print(f"[INIT] TEST_RANGE: {TEST_RANGE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INIT] Output directory ensured: {OUTPUT_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
print(f"[INIT] Random seeds fixed (SEED={SEED}), cudnn deterministic enabled")

def get_fixed_generator():
    print(f"  [FUNC] get_fixed_generator called, device={device}, seed={SEED}")
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== 模型加载 =====================
print("=" * 60)
print("[DEBUG Init] Loading ControlNet & SD base pipelines")
print("[INIT] Loading pose ControlNet...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
print("[INIT] pose ControlNet loaded successfully")

print("[INIT] Loading depth ControlNet...")
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)
print("[INIT] depth ControlNet loaded successfully")

print("[INIT] Loading base SD pipeline...")
pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
print("[INIT] base SD pipeline loaded successfully")

print("[INIT] Loading pose-only ControlNet pipeline...")
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
print("[INIT] pose-only ControlNet pipeline loaded successfully")

print("[INIT] Loading pose+depth dual ControlNet pipeline...")
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)
print("[INIT] dual ControlNet pipeline loaded successfully")

print("[INIT] Setting UniPC scheduler for all pipelines...")
for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("[INIT] Scheduler configured for all 3 pipelines")

def preload_all_loras(pipe, char_to_filename):
    print(f"  [FUNC] preload_all_loras called, pipe type: {pipe.__class__.__name__}")
    print(f"  [FUNC] character count: {len(char_to_filename)}")
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        print(f"  [FUNC] loading LoRA: {char_id} -> {lora_path}")
        if not os.path.exists(lora_path):
            print(f"  [ERROR] LoRA file not found: {lora_path}")
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
        print(f"  [FUNC] LoRA '{char_id}' loaded as adapter")
    print(f"[DEBUG LoRA] Preloaded {len(char_to_filename)} character LoRAs")

# ===================== Token 解析 =====================
def get_person_token_indices(tokenizer, prompt, debug=False):
    print(f"  [FUNC] get_person_token_indices called")
    full_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    full_ids = full_inputs.input_ids[0].tolist()
    full_tokens = tokenizer.convert_ids_to_tokens(full_ids)
    clean = [t.replace("</w>", "").lower() for t in full_tokens]
    print(f"  [FUNC] total tokens: {len(clean)}")

    p1_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "1"), -1)
    p2_start = next((i for i in range(len(clean) - 1) if clean[i] == "person" and clean[i+1] == "2"), -1)
    print(f"  [FUNC] person1 start index: {p1_start}, person2 start index: {p2_start}")

    if p1_start == -1 or p2_start == -1:
        print(f"  [FUNC] person markers not found, using default ranges")
        return [list(range(5, 30)), list(range(35, 60))]

    p1_desc_start = p1_start + 2
    while p1_desc_start < p2_start and clean[p1_desc_start] in [":", " "]: p1_desc_start += 1
    p1_desc_end = p2_start
    while p1_desc_end > p1_desc_start and clean[p1_desc_end-1] in [".", " "]: p1_desc_end -= 1

    p2_desc_start = p2_start + 2
    while p2_desc_start < len(clean) and clean[p2_desc_start] in [":", " "]: p2_desc_start += 1
    p2_desc_end = next((i for i in range(p2_desc_start, len(clean)) if clean[i] == "."), len(clean))

    print(f"  [FUNC] person1 token range: [{p1_desc_start}, {p1_desc_end})")
    print(f"  [FUNC] person2 token range: [{p2_desc_start}, {p2_desc_end})")
    return [list(range(p1_desc_start, p1_desc_end)), list(range(p2_desc_start, p2_desc_end))]

def process_mask(mask_path):
    print(f"  [FUNC] process_mask called, path: {mask_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"  [FUNC] mask loaded, shape: {mask.shape}")
    
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    print(f"  [FUNC] mask1 (value=128) pixels: {np.count_nonzero(mask1)}")
    print(f"  [FUNC] mask2 (value=255) pixels: {np.count_nonzero(mask2)}")
    
    kernel = np.ones((5, 5), np.uint8)
    print(f"  [FUNC] applying dilation (kernel=5x5, iterations=1)")
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    
    print(f"  [FUNC] applying Gaussian blur (kernel=9x9, sigma=0.8)")
    mask1 = cv2.GaussianBlur(mask1, (9, 9), sigmaX=0.8)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), sigmaX=0.8)
    
    result = [Image.fromarray(mask1), Image.fromarray(mask2)]
    print(f"  [FUNC] process_mask done, returned 2 PIL masks")
    return result

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
        print(f"  [FUNC] IntegratedMultiRoleProcessor init")
        print(f"  [FUNC] char_ids: {char_ids}, weights: {weights}")
        print(f"  [FUNC] enable_cross_mask: {enable_cross_mask}, enable_regional_lora: {enable_regional_lora}")
        print(f"  [FUNC] penalty: {penalty}, boost: {boost}")
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
    print("  [FUNC] clear_gpu: running gc.collect() and torch.cuda.empty_cache()")
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("  [FUNC] clear_gpu done")

def reset_print_flags():
    print("  [FUNC] reset_print_flags called")
    for func in [compute_regional_lora_linear, compute_text_regional_lora_linear]:
        for attr in list(dir(func)):
            if attr.endswith("_printed"):
                delattr(func, attr)
    print("  [FUNC] reset_print_flags done")

# ===================== 主生成逻辑 =====================
def run_synthetic():
    print("\n" + "=" * 60)
    print("[MAIN] run_synthetic started")
    print("=" * 60)
    
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    print(f"[MAIN] Loading configs from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    print(f"[MAIN] Total configs in file: {len(configs)}")
    
    run_list = configs[TEST_RANGE[0]:TEST_RANGE[1]]
    print(f"[MAIN] Running samples {TEST_RANGE[0]} to {TEST_RANGE[1]} (total: {len(run_list)})")
    
    char_map = {"Sera": "LoRA_Sera", "TogaHimiko": "TogaHimiko-01", "MouriRan": "Mouri", "Byakuya": "Byakuya"}
    print(f"[MAIN] Character to LoRA file mapping: {char_map}")

    print("[MAIN] Preloading LoRAs into all pipelines...")
    for p in [pipe_base, pipe_pose, pipe_both]:
        print(f"[MAIN] Preloading into {p.__class__.__name__}...")
        preload_all_loras(p, char_map)
    print("[MAIN] All LoRAs preloaded")

    for cfg_idx, cfg in enumerate(run_list):
        print(f"\n{'='*60}")
        print(f"[MAIN] Processing sample {cfg_idx+1}/{len(run_list)}")
        print(f"{'='*60}")
        
        sid = cfg["sample_id"]
        print(f"[MAIN] sample_id: {sid}")
        
        prompt = re.sub(r"<lora:[^>]+>", "", cfg["prompt"])
        neg_prompt = cfg["negative_prompt"]
        c1, c2 = cfg["characters"]
        print(f"[MAIN] characters: {c1} vs {c2}")
        print(f"[MAIN] original prompt length: {len(prompt)} chars")
        
          
        print("[MAIN] Checking and adding view directives...")
        if "front view" not in prompt.lower():
            prompt = f"front view, {prompt}"
            print("[MAIN] added 'front view' to prompt (prepended)")
        if "back view" not in neg_prompt.lower():
            neg_prompt = f"{neg_prompt}, back view"
            print("[MAIN] added 'back view' to negative prompt")
            
        print(f"\n[DEBUG Main] Modified Prompt: {prompt}")
        print(f"[DEBUG Main] Modified Neg: {neg_prompt}")
        
        print("[MAIN] Loading control images...")
        pose_path = os.path.join(SYNTHETIC_DATA, "poses", f"{sid}.png")
        depth_path = os.path.join(SYNTHETIC_DATA, "depths", f"{sid}.png")
        mask_file = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")
        print(f"[MAIN] pose image: {pose_path}")
        print(f"[MAIN] depth image: {depth_path}")
        print(f"[MAIN] mask file: {mask_file}")
        
        pose_img = Image.open(pose_path).convert("RGB")
        depth_img = Image.open(depth_path).convert("RGB")
        print(f"[MAIN] pose image size: {pose_img.size}, depth image size: {depth_img.size}")

        print("[MAIN] Computing person token indices...")
        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt)
        print(f"[MAIN] token_ranges lengths: {[len(r) for r in token_ranges]}")
        
        print("[MAIN] Processing mask pair...")
        mask_pair = process_mask(mask_file)
        
        # =============== 基线 1-3 ===============
        print("\n" + "-" * 40)
        print("[MAIN] Setting up baseline pipelines (1-3)")
        print("-" * 40)
        
        for p in [pipe_base, pipe_pose, pipe_both]:
            pipe_name = p.__class__.__name__
            print(f"[MAIN] Configuring {pipe_name}...")
            
            #     先全局设置适配器，再单独关闭 text_encoder 的 LoRA
            # 避免 set_adapters 重新激活 text_encoder LoRA，确保仅 unet 生效
            print(f"[MAIN]   setting adapters: {c1}, {c2} (weights 0.8, 0.8)")
            p.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
            
            if hasattr(p, "text_encoder") and p.text_encoder is not None:
                try:
                    p.text_encoder.set_adapter([]) 
                    print(f"[DEBUG LoRA] Text Encoder LoRA DEACTIVATED for {pipe_name}")
                except Exception as e:
                    print(f"[WARNING] Could not disable text encoder LoRA: {e}")
            
            print(f"[MAIN]   resetting UNet attention processor to default AttnProcessor")
            p.unet.set_attn_processor(AttnProcessor())

        # Baseline 1
        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[0]} (baseline1: base SD, no ControlNet)")
        result = pipe_base(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            generator=get_fixed_generator(), 
            num_inference_steps=25, 
            guidance_scale=7.5,
            progress_bar=False
        )
        save_path = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[0]}.png")
        result.images[0].save(save_path)
        print(f"[-] Saved to: {save_path}")
        clear_gpu()

        # Baseline 2
        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[1]} (baseline2: pose ControlNet only)")
        result = pipe_pose(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=pose_img, 
            controlnet_conditioning_scale=0.7, 
            generator=get_fixed_generator(), 
            num_inference_steps=25, 
            guidance_scale=7.5,
            progress_bar=False
        )
        save_path = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[1]}.png")
        result.images[0].save(save_path)
        print(f"[-] Saved to: {save_path}")
        clear_gpu()

        # Baseline 3
        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[2]} (baseline3: pose+depth ControlNet)")
        result = pipe_both(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.8, 0.4], 
            generator=get_fixed_generator(), 
            num_inference_steps=25, 
            guidance_scale=7.5,
            progress_bar=False
        )
        save_path = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[2]}.png")
        result.images[0].save(save_path)
        print(f"[-] Saved to: {save_path}")
        clear_gpu()

        # =============== 隔离组 ===============
        print("\n" + "-" * 40)
        print("[MAIN] Setting up isolation group (baseline4 + method)")
        print("-" * 40)
        
        print("[MAIN] Clearing all adapters from UNet...")
        pipe_both.unet.set_adapters([])
        if hasattr(pipe_both, "text_encoder") and pipe_both.text_encoder is not None:
            try:
                pipe_both.text_encoder.set_adapter([])
                print("[MAIN] Text encoder adapters also cleared")
            except:
                print("[MAIN] Text encoder adapter clear skipped (not supported)")
                pass
        
        # Baseline 4
        reset_print_flags()
        print(f"\n[*] Generating {sid}{METHOD_SUFFIX[3]} (Regional LoRA Only, no cross-mask)")
        pipe_both.unet.set_attn_processor(IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=False, enable_regional_lora=True
        ))
        print("[*] IntegratedMultiRoleProcessor installed (regional_lora=True, cross_mask=False)")
        
        result = pipe_both(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.8, 0.4], 
            generator=get_fixed_generator(), 
            num_inference_steps=25, 
            guidance_scale=7.5,
            progress_bar=False
        )
        save_path = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[3]}.png")
        result.images[0].save(save_path)
        print(f"[*] Saved to: {save_path}")
        clear_gpu()

        # Method
        reset_print_flags()
        print(f"\n[+] Generating {sid}{METHOD_SUFFIX[4]} (Cross-Attention Mask + Regional LoRA)")
        pipe_both.unet.set_attn_processor(IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=True, enable_regional_lora=True,
            penalty=15.0, boost=2.0
        ))
        print("[+] IntegratedMultiRoleProcessor installed (regional_lora=True, cross_mask=True, penalty=15, boost=2)")
        
        #     统一 ControlNet 权重与基线3一致，保证单一变量，实验对比公平
        print("[+] ControlNet scales: pose=0.8, depth=0.4 (same as baseline3)")
        result = pipe_both(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.8, 0.4], 
            generator=get_fixed_generator(), 
            num_inference_steps=25, 
            guidance_scale=7.5,
            progress_bar=False
        )
        save_path = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[4]}.png")
        result.images[0].save(save_path)
        print(f"[+] Saved to: {save_path}")
        
        print("[MAIN] Resetting UNet attention processor to default")
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu()

    print("\n" + "=" * 60)
    print(" All experiments generated successfully.")
    print("=" * 60)

if __name__ == "__main__":
    print("[ENTRY] Script started")
    try:
        run_synthetic()
    except Exception as e:
        print("[ERROR] Exception occurred during execution:")
        import traceback; traceback.print_exc()
        clear_gpu()
        raise