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

# ===================== Global Config =====================
SYNTHETIC_DATA = "synthetic_test_dataset"
OUTPUT_DIR = "synthetic_results"      
LORA_WEIGHTS_DIR = "./lora_weights"
IMAGE_SIZE = 512
SEED = 42

TEST_RANGE = [0, 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Debug counter: count layers successfully applied regional LoRA
lora_hit_count = {"spatial": 0, "text": 0}

def get_fixed_generator():
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== Model Loading =====================
print("=" * 60)
print(" [+] Loading ControlNet & SD base pipelines")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_both = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=[controlnet_pose, controlnet_depth], 
    torch_dtype=torch.float16, 
    safety_checker=None
).to(device)

pipe_both.scheduler = UniPCMultistepScheduler.from_config(pipe_both.scheduler.config)

def preload_all_loras(pipe, char_to_filename):
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
    print(f"Preloaded {len(char_to_filename)} character LoRAs")

# ========== Fix: Token parsing compatible with merged tokens containing dot ==========
def get_person_token_indices(tokenizer, prompt, debug=False):
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower().strip(":") for t in tokens]

    p1_start = p1_end = p2_start = p2_end = -1

    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person1" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "1"):
            p1_start = i
            break

    if p1_start != -1:
        for j in range(p1_start + 1, len(clean_tokens)):
            is_person2 = (
                clean_tokens[j] == "person2"
                or (clean_tokens[j] == "person" and j + 1 < len(clean_tokens) and clean_tokens[j+1] == "2")
            )
            if is_person2 or clean_tokens[j] == "<|endoftext|>":
                p1_end = j
                break

    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person2" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "2"):
            p2_start = i
            break

    if p2_start != -1:
        for j in range(p2_start + 1, len(clean_tokens)):
            tok = clean_tokens[j]
            if tok == "<|endoftext|>" or "." in tok:
                p2_end = j
                break

    if p1_start != -1 and p1_end != -1 and p2_start != -1 and p2_end != -1 and p2_start >= p1_end:
        result = [list(range(p1_start, p1_end)), list(range(p2_start, p2_end))]
        if debug:
            print(f"✅ Token parse success: person1 token range {result[0]}, person2 token range {result[1]}")
        return result

    if debug:
        print("⚠️ Token parse failed, fallback range will be used!")
    return [list(range(5, 20)), list(range(25, 45))]

# ========== Mask order: person1(left), person2(right) ==========
def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    mask_p1 = cv2.dilate((mask == 128).astype(np.uint8) * 255, kernel, iterations=1)
    mask_p2 = cv2.dilate((mask == 255).astype(np.uint8) * 255, kernel, iterations=1)
    mask_p1 = cv2.GaussianBlur(mask_p1, (9, 9), sigmaX=1.5)
    mask_p2 = cv2.GaussianBlur(mask_p2, (9, 9), sigmaX=1.5)
    return [Image.fromarray(mask_p1), Image.fromarray(mask_p2)]

# ========== [Enhanced] Multi-compatible LoRA component extractor ==========
def _get_lora_components(layer, adapter_name):
    # Compatibility 1: PEFT standard LoRALinear (base_layer + lora_A/lora_B dict)
    if hasattr(layer, "base_layer") and hasattr(layer, "lora_A"):
        if isinstance(layer.lora_A, dict) and adapter_name in layer.lora_A:
            scaling = layer.scaling[adapter_name] if hasattr(layer, "scaling") and adapter_name in layer.scaling else 1.0
            return layer.base_layer, layer.lora_A[adapter_name], layer.lora_B[adapter_name], scaling
    
    # Compatibility 2: PEFT nested lora_layer structure
    if hasattr(layer, "lora_layer"):
        ll = layer.lora_layer
        if hasattr(ll, "lora_A") and isinstance(ll.lora_A, dict) and adapter_name in ll.lora_A:
            scaling = ll.scaling[adapter_name] if hasattr(ll, "scaling") and adapter_name in ll.scaling else 1.0
            return layer, ll.lora_A[adapter_name], ll.lora_B[adapter_name], scaling
        if hasattr(ll, "lora_A") and hasattr(ll.lora_A, "weight"):
            scaling = ll.scaling if hasattr(ll, "scaling") else 1.0
            return layer, ll.lora_A, ll.lora_B, scaling
    
    # Compatibility 3: diffusers native LoRA layers
    if hasattr(layer, "lora_A_weights") and adapter_name in layer.lora_A_weights:
        scaling = layer.scaling[adapter_name] if hasattr(layer, "scaling") else 1.0
        return layer, layer.lora_A_weights[adapter_name], layer.lora_B_weights[adapter_name], scaling

    return None

# ========== Spatial self-attention LoRA: mask only spatial layers, no LoRA on non-spatial layers ==========
def compute_regional_lora_linear(layer, x, masks, char_ids, weights, enable_regional=True):
    global lora_hit_count
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None or not enable_regional:
        return layer(x)

    lora_hit_count["spatial"] += 1
    base_layer, _, _, _ = lora_test
    batch_size = x.shape[0]
    cond_split = batch_size // 2

    x_uncond = x[:cond_split]
    x_cond = x[cond_split:]
    out_uncond = layer(x_uncond)  # Uncond branch raw output without LoRA
    base_out_cond = base_layer(x_cond)
    combined_delta = torch.zeros_like(base_out_cond)

    seq_len = x_cond.shape[1]
    spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None or weights[idx] == 0:
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx]

        if spatial_dim in [8, 16, 32, 64] and idx < len(masks):
            m_arr = np.array(masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
            mask_tensor = torch.tensor(m_arr, device=x.device, dtype=x.dtype).view(1, -1, 1)
            combined_delta += delta * mask_tensor
        else:
            pass  # No LoRA applied to non-spatial layers

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

# ========== Text cross-attention LoRA: apply only to matched token ranges ==========
def compute_text_regional_lora_linear(layer, x, token_indices, char_ids, weights):
    global lora_hit_count
    lora_test = _get_lora_components(layer, char_ids[0])
    if lora_test is None:
        return layer(x)

    lora_hit_count["text"] += 1
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
        if comp is None or weights[idx] == 0:
            continue
        _, lora_A, lora_B, scaling = comp
        delta = lora_B(lora_A(x_cond)) * scaling * weights[idx]

        token_mask = torch.zeros((1, x_cond.shape[1], 1), device=x.device, dtype=x.dtype)
        if token_indices and idx < len(token_indices):
            for tid in token_indices[idx]:
                if tid < x_cond.shape[1]:
                    token_mask[0, tid, 0] = 1.0
            combined_delta += delta * token_mask

    out_cond = base_out_cond + combined_delta
    return torch.cat([out_uncond, out_cond], dim=0)

# ========== Test Processor: Regional LoRA + Cross-Attention Mask Bias ==========
class TestRegionalLoRAProcessor(AttnProcessor):
    def __init__(self, masks, char_ids, weights, token_indices, penalty=25.0, boost=2.0):
        super().__init__()
        self.masks = masks
        self.char_ids = char_ids
        self.weights = weights
        self.token_indices = token_indices
        self.penalty = penalty
        self.boost = boost

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        spatial_dim = int(np.sqrt(seq_len)) if (int(np.sqrt(seq_len)) ** 2 == seq_len) else None

        q = compute_regional_lora_linear(attn.to_q, hidden_states, self.masks, self.char_ids, self.weights, True)

        if is_cross:
            k = compute_text_regional_lora_linear(attn.to_k, encoder_hidden_states, self.token_indices, self.char_ids, self.weights)
            v = compute_text_regional_lora_linear(attn.to_v, encoder_hidden_states, self.token_indices, self.char_ids, self.weights)
        else:
            k = compute_regional_lora_linear(attn.to_k, hidden_states, self.masks, self.char_ids, self.weights, True)
            v = compute_regional_lora_linear(attn.to_v, hidden_states, self.masks, self.char_ids, self.weights, True)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        attn_score = torch.bmm(q, k.transpose(-1, -2)) * attn.scale

        # Cross attention spatial bias: force spatial region only attend matched character tokens
        if is_cross and self.token_indices and spatial_dim in [8, 16, 32, 64]:
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

        hidden = compute_regional_lora_linear(attn.to_out[0], hidden, self.masks, self.char_ids, self.weights, True)
        hidden = attn.to_out[1](hidden)
        return hidden

def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== Main Test Pipeline =====================
def run_alignment_test():
    global lora_hit_count
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    start_idx, end_idx = TEST_RANGE
    run_list = configs[start_idx:end_idx]

    char_map = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
    }
    preload_all_loras(pipe_both, char_map)

    for cfg in run_list:
        sid = cfg["sample_id"]
        prompt = re.sub(r"<lora:[^>]+>", "", cfg["prompt"])
        neg_prompt = cfg["negative_prompt"]
        c1, c2 = cfg["characters"]
        pose_img = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sid}.png")).convert("RGB")
        depth_img = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sid}.png")).convert("RGB")
        mask_file = os.path.join(SYNTHETIC_DATA, "masks", f"{sid}.png")

        mask_pair = process_mask(mask_file)
        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt, debug=True)

        # 1. Save mask preview
        mask_pair[0].save(os.path.join(OUTPUT_DIR, f"{sid}_mask1_p1_left.png"))
        mask_pair[1].save(os.path.join(OUTPUT_DIR, f"{sid}_mask2_p2_right.png"))
        print(f"\n[1/4] Mask preview saved: mask1=left person1, mask2=right person2")

        # 2. Baseline: global dual LoRA reference
        pipe_both.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
        pipe_both.unet.set_attn_processor(AttnProcessor())
        print(f"[2/4] Generating baseline reference: global dual LoRA")
        img_global = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(), 
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img_global.save(os.path.join(OUTPUT_DIR, f"{sid}_ref_global_both.png"))
        clear_gpu()

        # Unload global LoRA, reset counter
        pipe_both.set_adapters([])
        lora_hit_count = {"spatial": 0, "text": 0}

        # 3. Test A: Only left person1 apply Asuna LoRA
        print(f"[3/4] Generating Test A: Only left person1 loaded {c1} LoRA")
        pipe_both.unet.set_attn_processor(
            TestRegionalLoRAProcessor(
                masks=mask_pair, 
                char_ids=[c1, c2], 
                weights=[0.8, 0.0],
                token_indices=token_ranges
            )
        )
        img_p1_only = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(), 
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img_p1_only.save(os.path.join(OUTPUT_DIR, f"{sid}_test_p1_only.png"))
        print(f"   📊 Generation stats: spatial LoRA hit {lora_hit_count['spatial']} layers, text LoRA hit {lora_hit_count['text']} layers")
        clear_gpu()

        # Reset counter
        lora_hit_count = {"spatial": 0, "text": 0}

        # 4. Test B: Only right person2 apply Neferpitou LoRA
        print(f"[4/4] Generating Test B: Only right person2 loaded {c2} LoRA")
        pipe_both.unet.set_attn_processor(
            TestRegionalLoRAProcessor(
                masks=mask_pair, 
                char_ids=[c1, c2], 
                weights=[0.0, 0.8],
                token_indices=token_ranges
            )
        )
        img_p2_only = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt, 
            image=[pose_img, depth_img], 
            controlnet_conditioning_scale=[0.7, 0.5],
            generator=get_fixed_generator(), 
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img_p2_only.save(os.path.join(OUTPUT_DIR, f"{sid}_test_p2_only.png"))
        print(f"   📊 Generation stats: spatial LoRA hit {lora_hit_count['spatial']} layers, text LoRA hit {lora_hit_count['text']} layers")
        clear_gpu()

        pipe_both.unet.set_attn_processor(AttnProcessor())

    print("\n✅ Alignment test finished completely!")
    print("💡 If hit count >0: regional LoRA works, obvious character difference between left & right; If hit count=0: LoRA layer structure mismatched, need to adjust extract logic.")

if __name__ == "__main__":
    try:
        run_alignment_test()
    except Exception as e:
        clear_gpu()
        raise