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
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method", "_ablation1", "_ablation2"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
generator = torch.Generator(device).manual_seed(SEED)

# ===================== 加载模型 =====================
print("="*60)
print(" 正在并行加载 ControlNet 模型与计算流...")
controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16).to(device)
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe_both = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth], torch_dtype=torch.float16, safety_checker=None).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# =====================  LoRA 动态加载 =====================
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

# ===================== Token 定位（已修复接口签名与自动截断 Bug） =====================
def get_person_token_indices(tokenizer, prompt):
    """通过解析 'person1' 和 'person2' 的区间边界，精准隔离角色特征描述文本"""
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    p1_idx = [i for i, t in enumerate(clean_tokens) if "person1" in t or (t == "person" and i+1 < len(clean_tokens) and "1" in clean_tokens[i+1])]
    p2_idx = [i for i, t in enumerate(clean_tokens) if "person2" in t or (t == "person" and i+1 < len(clean_tokens) and "2" in clean_tokens[i+1])]
    
    if not p1_idx or not p2_idx:
        raise ValueError(f"Prompt 无法检出两段式结构标签：{prompt}")
        
    start_p1 = p1_idx[0]
    start_p2 = p2_idx[0]
    
    # person1 跨度终止于 person2 出现之前
    person1_token_ids = list(range(start_p1, start_p2))
    
    # 自动在包含 'scene' 的公共标记处截止，确保后方公共场景背景词汇不被掩码强制拦截
    end_p2 = len(clean_tokens) - 1
    for i in range(start_p2, len(clean_tokens)):
        if "scene" in clean_tokens[i] or clean_tokens[i] in ["<|endoftext|>", "", ".", ","]:
            end_p2 = i - 1
            break
    person2_token_ids = list(range(start_p2, end_p2 + 1))
    
    return [person1_token_ids, person2_token_ids]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

# ===================== 交叉注意力掩码拦截处理器 =====================
class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks):
        super().__init__()
        self.token_indices = token_indices
        self.masks = masks

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

        attn_scores = torch.bmm(query, key.transpose(-1, -2)) / attn.scale

        if not is_self_attn and self.token_indices:
            spatial_seq_len = attn_scores.shape[-2]
            spatial_size = int(np.sqrt(spatial_seq_len))
            
            for i, token_ids in enumerate(self.token_indices):
                mask_img = self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.NEAREST)
                m_arr = np.array(mask_img) / 255.0
                mask_tensor = torch.tensor(m_arr, device=query.device, dtype=query.dtype)
                
                # 计算边界外特征的负数惩罚项
                mask_neg = (1.0 - mask_tensor).view(-1) * -10000.0
                
                for token_idx in token_ids:
                    if token_idx >= attn_scores.shape[-1]:
                        continue
                    # 仅针对条件生成分支（Batch 维度的后半段，即正向提示词引导流）施加截断惩罚 [cite: 39, 40]
                    half_idx = attn_scores.shape[0] // 2
                    attn_scores[half_idx:, :, token_idx] += mask_neg.unsqueeze(0)

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

def apply_attention_mask(pipe, token_indices, masks):
    pipe.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks))

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 核心主流程 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    print(f"\n开始运行全量合成测试集实验，共 {len(configs)} 组角色排列组合样本...")

    char_to_filename = {
        "Asuna": "asuna_(stacia)-v1.5",
        "Neferpitou": "LoRA_Neferpitou",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    for idx, cfg in enumerate(configs):
        sample_id = cfg["sample_id"]
        prompt = cfg["prompt"]
        neg_prompt = cfg["negative_prompt"]

        char1_name = char_to_filename[cfg["characters"][0]]
        char2_name = char_to_filename[cfg["characters"][1]]

        prompt = re.sub(r"<lora:[^>]+>", "", prompt)
        print(f"\n[{idx+1}/{len(configs)}] 正在处理自动化流水线样本: {sample_id}")

        for pipe in [pipe_base, pipe_pose, pipe_both]:
            load_loras_for_pair(pipe, char1_name, char2_name)

        pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
        depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
        mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
        
        # 修复了旧版缺失传参导致的崩溃错误，当前只需要传入 2 个核心参数
        token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

        # 1. Baseline 1: 纯文本控制 [cite: 92]
        img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[0]}.png"))
        clear_gpu_memory()

        # 2. Baseline 2: 单 OpenPose 骨骼图控制 [cite: 95]
        img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[1]}.png"))
        clear_gpu_memory()

        # 3. Baseline 3: 双 ControlNet (Pose + Depth) [cite: 96]
        img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img3.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[2]}.png"))
        clear_gpu_memory()

        # 4. Method: 核心方案（双 ControlNet + 实例交叉注意力隔离掩码） [cite: 60]
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_both, token_indices, masks)
        img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
        pipe_both.unet.set_attn_processor(AttnProcessor()) # 还原默认处理器，防止权重污染
        clear_gpu_memory()

        # 5. Ablation 1: 消融组 1（双 ControlNet + 噪声随机掩码破坏）
        rand1 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        rand2 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        apply_attention_mask(pipe_both, token_indices, [rand1, rand2])
        img_ab1 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[4]}.png"))
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        # 6. Ablation 2: 消融组 2（单 OpenPose 骨骼 + 实例注意力隔离掩码）
        apply_attention_mask(pipe_pose, token_indices, masks)
        img_ab2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[5]}.png"))
        pipe_pose.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        print(f" [+] 样本 {sample_id} 的 6 个对比/消融实验数据全部离线落地完成。")

    print(f"\n自动化实验已全部顺利运行完成，结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        print(f"\n运行时遇到致命阻断错误: {str(e)}")
        clear_gpu_memory()
        raise