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

# 每次生成都重新实例化生成器，确保初始隐空间噪声 100% 对齐
def get_fixed_generator():
    g = torch.Generator(device)
    g.manual_seed(SEED)
    return g

# ===================== 模型加载 =====================
print("=" * 60)
print("[DEBUG Init] Loading ControlNet & SD base pipelines")
print(f"[DEBUG Init] Device: {device}")
print(f"[DEBUG Init] Torch version: {torch.__version__}")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)
print("[DEBUG Init] ControlNet models loaded")

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)
print("[DEBUG Init] All SD pipelines loaded")

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
print("[DEBUG Init] Scheduler set to UniPCMultistepScheduler")

def preload_all_loras(pipe, char_to_filename):
    print(f"\n[DEBUG LoRA] Start preloading LoRAs for pipeline...")
    for char_id, filename in char_to_filename.items():
        lora_path = os.path.join(LORA_WEIGHTS_DIR, f"{filename}.safetensors")
        print(f"[DEBUG LoRA] Loading {char_id} from: {lora_path}")
        if not os.path.exists(lora_path):
            print(f"[ERROR LoRA] File NOT FOUND: {lora_path}")
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")
        pipe.load_lora_weights(lora_path, adapter_name=char_id)
        print(f"[DEBUG LoRA] Successfully loaded adapter: {char_id}")
    print(f"[DEBUG LoRA] Preloaded {len(char_to_filename)} character LoRAs")
    print(f"[DEBUG LoRA] Current active adapters: {pipe.get_active_adapters()}")
    print(f"[DEBUG LoRA] All available adapters: {pipe.get_list_adapters()}")

# 增强型 BPE 文本拦截，确保特定角色的特征词被精准框定
def get_person_token_indices(tokenizer, prompt, debug=False):
    print("\n[DEBUG Token] === Start parsing token ranges ===")
    print(f"[DEBUG Token] Raw prompt: {prompt[:150]}..." if len(prompt)>150 else f"[DEBUG Token] Raw prompt: {prompt}")

    # ========== 修复1：优先通过 person1:/person2: 精准分割角色描述 ==========
    if "person1:" in prompt and "person2:" in prompt:
        print("[DEBUG Token] Using string-split parsing method")
        p1_str_start = prompt.find("person1:") + len("person1:")
        p2_str_pos = prompt.find("person2:")
        p2_str_start = p2_str_pos + len("person2:")

        # 精确截取两个角色的描述文本范围
        p1_desc = prompt[p1_str_start:p2_str_pos].strip()
        p2_desc = prompt[p2_str_start:].strip()
        print(f"[DEBUG Token] Extracted person1 desc: [{p1_desc}]")
        print(f"[DEBUG Token] Extracted person2 desc: [{p2_desc}]")

        # 分词得到token数量
        p1_tokens = tokenizer.tokenize(p1_desc)
        p2_tokens = tokenizer.tokenize(p2_desc)
        print(f"[DEBUG Token] person1 token count: {len(p1_tokens)}, tokens: {p1_tokens[:10]}...")
        print(f"[DEBUG Token] person2 token count: {len(p2_tokens)}, tokens: {p2_tokens[:10]}...")

        # 计算token索引范围（起始偏移对齐CLIP BPE的前缀token位置）
        base_offset = 5
        p1_range = list(range(base_offset, base_offset + len(p1_tokens)))
        p2_range = list(range(base_offset + len(p1_tokens), base_offset + len(p1_tokens) + len(p2_tokens)))

        print(f"[DEBUG Token] Calculated person1 range: {p1_range[0]} ~ {p1_range[-1]}")
        print(f"[DEBUG Token] Calculated person2 range: {p2_range[0]} ~ {p2_range[-1]}")

        # 校验：和完整编码对比
        full_ids = tokenizer.encode(prompt)
        print(f"[DEBUG Token] Full prompt total tokens: {len(full_ids)}")
        if p2_range[-1] >= len(full_ids):
            print(f"[WARNING Token] Person2 range exceeds total token length! Range end: {p2_range[-1]}, total: {len(full_ids)}")

        return [p1_range, p2_range]

    # ========== 原始回退逻辑 ==========
    print("[DEBUG Token] String split failed, falling back to token scan method")
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower().strip(":").strip() for t in tokens]

    p1_start = p1_end = p2_start = p2_end = -1

    # 查找 person1 起点
    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person1" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "1"):
            p1_start = i
            print(f"[DEBUG Token] Found person1 start at index {i}: {tok}")
            break

    # 查找 person1 终点（遇到 person2 停止）
    if p1_start != -1:
        for j in range(p1_start + 1, len(clean_tokens)):
            is_person2 = (
                clean_tokens[j] == "person2"
                or (clean_tokens[j] == "person" and j + 1 < len(clean_tokens) and clean_tokens[j+1] == "2")
            )
            if is_person2 or clean_tokens[j] == "<|endoftext|>":
                p1_end = j
                print(f"[DEBUG Token] Found person1 end at index {j}: {clean_tokens[j]}")
                break

    # 查找 person2 起点
    for i in range(len(clean_tokens)):
        tok = clean_tokens[i]
        if tok == "person2" or (tok == "person" and i + 1 < len(clean_tokens) and clean_tokens[i+1] == "2"):
            p2_start = i
            print(f"[DEBUG Token] Found person2 start at index {i}: {tok}")
            break

    # 查找 person2 终点（遇到句号停止）
    if p2_start != -1:
        for j in range(p2_start + 1, len(clean_tokens)):
            if clean_tokens[j] == "." or clean_tokens[j] == "<|endoftext|>":
                p2_end = j
                print(f"[DEBUG Token] Found person2 end at index {j}: {clean_tokens[j]}")
                break

    if p1_start != -1 and p1_end != -1 and p2_start != -1 and p2_end != -1 and p2_start >= p1_end:
        print(f"[DEBUG Token] Scan result: p1 [{p1_start}:{p1_end}], p2 [{p2_start}:{p2_end}]")
        return [list(range(p1_start, p1_end)), list(range(p2_start, p2_end))]

    print("[WARNING Token] All parsing failed, using hardcoded fallback range [5:20] and [25:45]")
    return [list(range(5, 20)), list(range(25, 45))]

def process_mask(mask_path):
    print(f"\n[DEBUG Mask] Processing mask: {mask_path}")
    # 直接以灰度图模式读取（单通道，像素值为 0, 128, 255）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"[ERROR Mask] Failed to read mask image!")
        raise FileNotFoundError(f"Mask not found or invalid: {mask_path}")

    print(f"[DEBUG Mask] Raw mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"[DEBUG Mask] Unique pixel values in raw mask: {np.unique(mask)}")

    # 核心修复：直接根据灰度值分割，不再寻找 Alpha 通道！
    # 128 = 角色1, 255 = 角色2
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255

    print(f"[DEBUG Mask] mask1 (128) non-zero pixels: {np.count_nonzero(mask1)}")
    print(f"[DEBUG Mask] mask2 (255) non-zero pixels: {np.count_nonzero(mask2)}")

    if np.count_nonzero(mask1) == 0:
        print("[WARNING Mask] mask1 (value 128) is EMPTY! Check your mask generation code.")
    if np.count_nonzero(mask2) == 0:
        print("[WARNING Mask] mask2 (value 255) is EMPTY! Check your mask generation code.")

    # 形态学膨胀，填补骨骼线之间的缝隙
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)
    print("[DEBUG Mask] Applied dilation")

    # 高斯模糊，柔化边缘
    mask1 = cv2.GaussianBlur(mask1, (9, 9), sigmaX=0.8)
    mask2 = cv2.GaussianBlur(mask2, (9, 9), sigmaX=0.8)
    print("[DEBUG Mask] Applied Gaussian blur")

    print(f"[DEBUG Mask] Final mask1 value range: [{mask1.min()}, {mask1.max()}]")
    print(f"[DEBUG Mask] Final mask2 value range: [{mask2.min()}, {mask2.max()}]")

    return [Image.fromarray(mask1), Image.fromarray(mask2)]

def _get_lora_components(layer, adapter_name):
    # 静默函数，不打印，避免日志爆炸；外层调用处打印
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

    # 只在第一个有效空间层打印一次，避免刷屏
    if spatial_dim in [8, 16, 32, 64] and not hasattr(compute_regional_lora_linear, "_printed"):
        compute_regional_lora_linear._printed = True
        print(f"[DEBUG Regional LoRA] Applying spatial LoRA at resolution {spatial_dim}x{spatial_dim}")
        print(f"[DEBUG Regional LoRA] x_cond shape: {x_cond.shape}")

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            print(f"[WARNING Regional LoRA] Cannot get components for {cid}")
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

    if not hasattr(compute_text_regional_lora_linear, "_printed"):
        compute_text_regional_lora_linear._printed = True
        print(f"[DEBUG Text LoRA] Applying token-wise LoRA, x_cond shape: {x_cond.shape}")

    for idx, cid in enumerate(char_ids):
        comp = _get_lora_components(layer, cid)
        if comp is None:
            print(f"[WARNING Text LoRA] Cannot get components for {cid}")
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

# ===================== 特征隔离系统 =====================
class IntegratedMultiRoleProcessor(AttnProcessor):
    def __init__(
        self, token_indices, masks, char_ids, weights=[0.8, 0.8],
        enable_cross_mask=True, enable_regional_lora=False,
        penalty=25.0, boost=1.5
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

        # 初始化打印
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

        # 只在第一次进入时打印一次关键信息
        if not hasattr(self, "_call_printed"):
            self._call_printed = True
            print(f"\n[DEBUG Processor Call] First forward pass")
            print(f"[DEBUG Processor Call] is_cross_attention: {is_cross}")
            print(f"[DEBUG Processor Call] hidden_states shape: {hidden_states.shape}")
            print(f"[DEBUG Processor Call] spatial_dim detected: {spatial_dim}")
            if is_cross:
                print(f"[DEBUG Processor Call] encoder_hidden_states shape: {encoder_hidden_states.shape}")

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
            if not hasattr(self, "_mask_applied_printed"):
                self._mask_applied_printed = True
                print(f"[DEBUG Processor] Applying cross-attention bias mask at resolution {spatial_dim}")

            cond_split = attn_score.shape[0] // 2
            for idx, token_ids in enumerate(self.token_indices):
                if idx >= len(self.masks): break
                m_arr = np.array(self.masks[idx].resize((spatial_dim, spatial_dim), Image.Resampling.BILINEAR)) / 255.0
                mask_vec = torch.tensor(m_arr, device=q.device, dtype=q.dtype).view(-1)
                bias = mask_vec * self.boost - (1.0 - mask_vec) * self.penalty

                if not hasattr(self, f"_bias_stats_{idx}"):
                    setattr(self, f"_bias_stats_{idx}", True)
                    print(f"[DEBUG Processor] Char {idx} bias stats: min={bias.min():.2f}, max={bias.max():.2f}, mean={bias.mean():.2f}")

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
    if torch.cuda.is_available():
        print(f"[DEBUG GPU] Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB, reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

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

    # 角色与本地LoRA文件名映射
    char_map = {
        "Sera": "LoRA_Sera",
        "TogaHimiko": "TogaHimiko-01",
        "MouriRan": "Mouri",
        "Byakuya": "Byakuya",
    }
    print(f"[DEBUG Main] Character map: {char_map}")

    for p in [pipe_base, pipe_pose, pipe_both]:
        preload_all_loras(p, char_map)

    # ========== 验证全局LoRA预加载状态 ==========
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

        # 重置打印标记，每个样本重新打印一次关键信息
        for func in [compute_regional_lora_linear, compute_text_regional_lora_linear]:
            if hasattr(func, "_printed"):
                delattr(func, "_printed")

        token_ranges = get_person_token_indices(pipe_both.tokenizer, prompt, debug=True)
        mask_pair = process_mask(mask_file)

        # =============== 基线测试（应用全局污染 LoRA） ===============
        print("\n--- Switching to GLOBAL LoRA mode (baselines 1-3) ---")
        for p in [pipe_base, pipe_pose, pipe_both]:
            p.enable_lora()
            p.set_adapters([c1, c2], adapter_weights=[0.8, 0.8])
            p.unet.set_attn_processor(AttnProcessor())

        print(f"[DEBUG Main] Baseline LoRA status - active adapters: {pipe_both.get_active_adapters()}")
        print(f"[DEBUG Main] Attn processor type: {type(pipe_both.unet.attn_processors).__name__}")

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[0]} (Baseline 1: Pure text)")
        img1 = pipe_base(
            prompt=prompt, negative_prompt=neg_prompt,
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save1 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[0]}.png")
        img1.save(save1)
        print(f"[DEBUG Main] Saved to: {save1}")
        clear_gpu()

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[1]} (Baseline 2: Pose only)")
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

        print(f"\n[-] Generating {sid}{METHOD_SUFFIX[2]} (Baseline 3: Pose + Depth)")
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

        # =============== 隔离机制组（卸载全局 LoRA，交由底层拦截器分发） ===============
        print("\n--- Switching to REGIONAL LoRA mode (baseline4 & method) ---")
        pipe_both.set_adapters([])
        pipe_both.disable_lora()
        print(f"[DEBUG Main] After disable - active adapters: {pipe_both.get_active_adapters()}")
        print(f"[DEBUG Main] LoRA enabled status: {pipe_both.get_active_adapters() != []}")

        # Baseline 4 ：仅区域 LoRA 控制，关闭交叉注意力惩罚
        print(f"\n[*] Generating {sid}{METHOD_SUFFIX[3]} (Ablation: Regional LoRA Only)")
        processor4 = IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=False, enable_regional_lora=True
        )
        pipe_both.unet.set_attn_processor(processor4)
        print(f"[DEBUG Main] Attn processor switched to: {type(processor4).__name__}")

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

        # Proposed Method：融合交叉注意力掩码与区域 LoRA 控制
        print(f"\n[+] Generating {sid}{METHOD_SUFFIX[4]} (Proposed: Cross-Attention Mask + Regional LoRA)")
        processor_method = IntegratedMultiRoleProcessor(
            token_indices=token_ranges, masks=mask_pair, char_ids=[c1, c2], weights=[0.8, 0.8],
            enable_cross_mask=True, enable_regional_lora=True,
            penalty=25.0, boost=1.5
        )
        pipe_both.unet.set_attn_processor(processor_method)
        print(f"[DEBUG Main] Attn processor switched to: {type(processor_method).__name__}")
        print(f"[DEBUG Main] ControlNet scales: pose=0.8, depth=0.4")

        img5 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt,
            image=[pose_img, depth_img],
            controlnet_conditioning_scale=[0.8, 0.4],
            generator=get_fixed_generator(),
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        save5 = os.path.join(OUTPUT_DIR, f"{sid}{METHOD_SUFFIX[4]}.png")
        img5.save(save5)
        print(f"[DEBUG Main] Saved to: {save5}")

        # 恢复默认处理器
        pipe_both.unet.set_attn_processor(AttnProcessor())
        print("[DEBUG Main] Restored default AttnProcessor")
        clear_gpu()

    print("\n" + "="*60)
    print(" All experiments generated successfully with high-fidelity de-bleeding setup.")
    print("="*60)

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        print(f"\n[FATAL ERROR] Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu()
        raise