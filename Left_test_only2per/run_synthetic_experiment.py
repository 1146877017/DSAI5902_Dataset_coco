import os
import json
import torch
import cv2
import numpy as np
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
os.makedirs(LORA_WEIGHTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator(device).manual_seed(SEED)

# ===================== 加载基础模型 =====================
print("="*60)
print("  正在加载 ControlNet 模型...")
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
).to(device)

print("\n  [1/3] 加载纯文本管道...")
pipe_base = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to(device)

print("\n  [2/3] 加载单 OpenPose 控制管道...")
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose,
    torch_dtype=torch.float16, safety_checker=None
).to(device)

print("\n  [3/3] 加载双 ControlNet 管道...")
pipe_both = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth],
    torch_dtype=torch.float16, safety_checker=None
).to(device)

for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# ===================== LoRA 动态加载函数 =====================
def load_loras_for_pair(pipe, char1_name, char2_name):
    """为当前管道卸载所有旧 LoRA，然后加载指定两个角色的 LoRA"""
    pipe.unload_lora_weights()
    lora_path1 = os.path.join(LORA_WEIGHTS_DIR, f"{char1_name}.safetensors")
    lora_path2 = os.path.join(LORA_WEIGHTS_DIR, f"{char2_name}.safetensors")
    if not os.path.exists(lora_path1):
        raise FileNotFoundError(f"缺少 LoRA 文件: {lora_path1}")
    if not os.path.exists(lora_path2):
        raise FileNotFoundError(f"缺少 LoRA 文件: {lora_path2}")
    pipe.load_lora_weights(lora_path1, adapter_name="char1")
    pipe.load_lora_weights(lora_path2, adapter_name="char2")
    pipe.set_adapters(["char1", "char2"], adapter_weights=[0.8, 0.8])

# ===================== 注意力掩码工具 =====================
def get_person_token_indices(tokenizer, prompt):
    inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    person1_start = person2_start = None
    for i in range(len(clean_tokens) - 2, -1, -1):
        if clean_tokens[i] == "person":
            next_token = clean_tokens[i+1].strip(" :.,")
            if next_token == "1" and person1_start is None:
                person1_start = i
            elif next_token == "2" and person2_start is None:
                person2_start = i
        if person1_start is not None and person2_start is not None:
            break
    if person1_start is None or person2_start is None:
        raise ValueError(f"未检测到 person1/person2 结构: {prompt}")
    person1_end = person2_start - 1
    while person1_end > person1_start and clean_tokens[person1_end] in [",", ":", "."]:
        person1_end -= 1
    person2_end = len(clean_tokens) - 1
    for i in range(person2_start + 2, len(clean_tokens)):
        if clean_tokens[i] in ["<|endoftext|>", ""]:
            person2_end = i - 1
            break
    return [list(range(person1_start, person1_end + 1)), list(range(person2_start, person2_end + 1))]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 128).astype(np.uint8) * 255
    mask2 = (mask == 255).astype(np.uint8) * 255
    mask1 = cv2.resize(mask1, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    mask2 = cv2.resize(mask2, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks):
        super().__init__()
        self.token_indices = token_indices   # 两个列表，每个列表包含多个 token id
        self.masks = masks                   # [PIL.Image, PIL.Image]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attn_scores = torch.bmm(query, key.transpose(-1, -2)) / attn.scale
        
        if self.token_indices and len(self.token_indices) > 0:
            # 获取当前层的空间分辨率（sqrt(空间序列长度)）
            spatial_seq_len = attn_scores.shape[-2]   # 
            spatial_size = int(np.sqrt(spatial_seq_len))
            if spatial_size * spatial_size != spatial_seq_len:
                # 
                spatial_size = int(np.sqrt(seq_len))  # 从 hidden_states 推测
                
            for i, token_ids in enumerate(self.token_indices):
                # 将当前角色的掩码 resize 到该层分辨率
                mask_img = self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.LANCZOS)
                mask = torch.tensor(np.array(mask_img), device=query.device, dtype=query.dtype) / 255.0
                mask_neg = (1 - mask) * -10000.0
                mask_neg_flat = mask_neg.view(-1)   # (spatial_size*spatial_size,)
                
                for token_idx in token_ids:
                    if token_idx >= attn_scores.shape[-1]:
                        continue
                    # 对当前 token 施加掩码：attn_scores[:, :, token_idx] 形状 (B*H, spatial_len)
                    attn_scores[:, :, token_idx] += mask_neg_flat.unsqueeze(0)
        
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
    torch.cuda.synchronize()

# ===================== 主流程 =====================
def run_synthetic():
    config_path = os.path.join(SYNTHETIC_DATA, "synthetic_configs.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    total_samples = len(configs)
    print(f"\n开始运行合成测试集实验，共 {total_samples} 组样本")
    print(f"每组包含 6 个对比/消融实验")

    # 预先建立角色名到文件名的映射
    char_to_filename = {
        "Asuna": "asuna_(stacia)-v1.5",
        "YamasakiAnzu": "gantzyamasakianzu",
        "Vanilla": "super-vanilla-newlora-ver1-p",
        "TogaHimiko": "TogaHimiko-01",
        "OchacoUraraka": "OchacoUraraka-01",
    }

    for idx, cfg in enumerate(configs):
        sample_id = cfg["sample_id"]
        scene = cfg["scene"]
        prompt = cfg["prompt"]
        neg_prompt = cfg["negative_prompt"]
        char1_name = char_to_filename[cfg["characters"][0]]
        char2_name = char_to_filename[cfg["characters"][1]]

        print(f"\n[{idx+1}/{total_samples}] 处理样本: {sample_id}")

        # 动态加载当前配对的两个 LoRA 到所有管道
        for pipe in [pipe_base, pipe_pose, pipe_both]:
            load_loras_for_pair(pipe, char1_name, char2_name)

        # 读取控制图
        pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{sample_id}.png")).convert("RGB")
        depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{sample_id}.png")).convert("RGB")
        mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{sample_id}.png")
        token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

        # 1. Baseline1 纯文本
        img1 = pipe_base(prompt=prompt, negative_prompt=neg_prompt,
                         generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[0]}.png"))
        del img1; clear_gpu_memory()

        # 2. Baseline2 单 OpenPose
        img2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose,
                         controlnet_conditioning_scale=1.0, generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[1]}.png"))
        del img2; clear_gpu_memory()

        # 3. Baseline3 双 ControlNet
        img3 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                         controlnet_conditioning_scale=[1.0, 1.0], generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img3.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[2]}.png"))
        del img3; clear_gpu_memory()

        # 4. Method 双 ControlNet + 注意力掩码
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_both, token_indices, masks)
        img4 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                         controlnet_conditioning_scale=[1.0, 1.0], generator=generator,
                         num_inference_steps=25, guidance_scale=7.5).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[3]}.png"))
        del img4
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        # 5. Ablation1 双 ControlNet + 随机掩码
        rand1 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        rand2 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        apply_attention_mask(pipe_both, token_indices, [rand1, rand2])
        img_ab1 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
                            generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab1.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[4]}.png"))
        del img_ab1
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        # 6. Ablation2 单 OpenPose + 实例掩码
        apply_attention_mask(pipe_pose, token_indices, masks)
        img_ab2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose,
                            generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab2.save(os.path.join(OUTPUT_DIR, f"{sample_id}{METHOD_SUFFIX[5]}.png"))
        del img_ab2
        pipe_pose.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        print(f"   样本 {sample_id} 所有实验组生成完成！")

    print("\n全量合成测试集实验完成！")
    print(f"结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    try:
        run_synthetic()
    except Exception as e:
        print(f"\n实验中断错误: {str(e)}")
        clear_gpu_memory()
        raise