import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.models.attention_processor import AttnProcessor

# ===================== 配置 =====================
# 合成数据集路径
SYNTHETIC_DATA = "synthetic_test_dataset"
# 输出目录
OUTPUT_DIR = "synthetic_results"
# 固定尺寸
IMAGE_SIZE = 512
# 种子（保证复现）
SEED = 42
# 保存后缀
METHOD_SUFFIX = ["_baseline1", "_baseline2", "_baseline3", "_method", "_ablation1", "_ablation2"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.manual_seed(SEED)

# ===================== 加载模型 =====================
print(" 加载 ControlNet 模型...")
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
).to(device)

# 1. 纯文本 Baseline1
pipe_base = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose,  
    torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe_base.controlnet = None

# 2. OpenPose Baseline2
pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose,
    torch_dtype=torch.float16, safety_checker=None
).to(device)

# 3. OpenPose+Depth Baseline3 & method
pipe_both = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_pose, controlnet_depth],
    torch_dtype=torch.float16, safety_checker=None
).to(device)

# 调度器
for pipe in [pipe_base, pipe_pose, pipe_both]:
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# ===================== 注意力掩码工具函数 =====================
def get_person_token_indices(tokenizer, prompt):
    tokens = tokenizer.encode(prompt)
    indices = []
    for i, t in enumerate(tokens):
        if t in [tokenizer.encode("person1")[1], tokenizer.encode("person2")[1]]:
            indices.append(i)
    return indices[:2]

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = (mask == 1).astype(np.uint8) * 255
    mask2 = (mask == 2).astype(np.uint8) * 255
    mask1 = cv2.resize(mask1, (IMAGE_SIZE, IMAGE_SIZE))
    mask2 = cv2.resize(mask2, (IMAGE_SIZE, IMAGE_SIZE))
    return [Image.fromarray(mask1), Image.fromarray(mask2)]

class AttentionMaskProcessor(AttnProcessor):
    def __init__(self, token_indices, masks):
        super().__init__()
        self.token_indices = token_indices
        self.masks = masks  # [person1_mask, person2_mask]，shape [H, W]
        self.H, self.W = masks[0].size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_scores = torch.bmm(query, key.transpose(-1, -2)) / attn.scale

        # 给非目标区域加-10000，屏蔽无关区域
        if self.token_indices is not None and len(self.token_indices) > 0:
            seq_len = attention_scores.shape[-1]
            spatial_size = int(seq_len ** 0.5) if int(seq_len ** 0.5)**2 == seq_len else int((seq_len/2)**0.5)
            if spatial_size**2 != seq_len:
                spatial_size = self.H // 8  # 适配ControlNet的下采样尺寸
            
            for i, token_idx in enumerate(self.token_indices):
                if token_idx >= attention_scores.shape[-1]:
                    continue
                # 处理掩码，resize到和注意力图一样的尺寸
                mask = self.masks[i].resize((spatial_size, spatial_size), Image.Resampling.LANCZOS)
                mask = torch.tensor(np.array(mask)).float().to(query.device)
                mask = (mask / 255.0).view(-1)  # 展平为[spatial_size*spatial_size]
                # 给非目标区域加-10000，softmax后趋近于0
                attention_scores[:, :, token_idx] += (1 - mask) * -10000

        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 输出投影
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def apply_attention_mask(pipe, token_indices, masks):
    pipe.unet.set_attn_processor(
        AttentionMaskProcessor(token_indices, masks)
    )

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# ===================== 主流程 =====================
def run_synthetic():
    # 加载合成数据集配置
    with open(os.path.join(SYNTHETIC_DATA, "synthetic_configs.json"), "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    print(f"\n 开始运行合成测试集实验，共 {len(configs)} 个场景")

    for cfg in configs:
        scene_name = cfg["scene"]
        prompt = cfg["prompt"]
        neg_prompt = cfg["negative_prompt"]
        
        print(f"\n===== 生成场景：{scene_name} =====")

        # 读取控制图
        pose = Image.open(os.path.join(SYNTHETIC_DATA, "poses", f"{scene_name}.jpg")).convert("RGB")
        depth = Image.open(os.path.join(SYNTHETIC_DATA, "depths", f"{scene_name}.jpg")).convert("RGB")
        mask_path = os.path.join(SYNTHETIC_DATA, "masks", f"{scene_name}.png")
        token_indices = get_person_token_indices(pipe_both.tokenizer, prompt)

        # ---------------------- 1. Baseline1  ----------------------
        img1 = pipe_base(
            prompt=prompt, negative_prompt=neg_prompt,
            generator=generator, num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img1.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[0]}.png"))
        del img1; clear_gpu_memory()

        # ---------------------- 2. Baseline2 OpenPose ----------------------
        img2 = pipe_pose(
            prompt=prompt, negative_prompt=neg_prompt, image=pose,
            controlnet_conditioning_scale=1.0, generator=generator,
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img2.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[1]}.png"))
        del img2; clear_gpu_memory()

        # ---------------------- 3. Baseline3 双ControlNet ----------------------
        img3 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
            controlnet_conditioning_scale=[1.0, 1.0], generator=generator,
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img3.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[2]}.png"))
        del img3; clear_gpu_memory()

        # ---------------------- 4. method 注意力掩码+双ControlNet ----------------------
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_both, token_indices, masks)
        img4 = pipe_both(
            prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth],
            controlnet_conditioning_scale=[1.0, 1.0], generator=generator,
            num_inference_steps=25, guidance_scale=7.5
        ).images[0]
        img4.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[3]}.png"))
        del img4
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        # --- Ablation1：双ControlNet + 随机掩码 ---
        random_mask1 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        random_mask2 = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        apply_attention_mask(pipe_both, token_indices, [random_mask1, random_mask2])
        img_ab1 = pipe_both(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab1.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[4]}.png"))
        del img_ab1
        pipe_both.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        # --- Ablation2：仅OpenPose + 实例掩码 ---
        apply_attention_mask(pipe_pose, token_indices, masks)
        img_ab2 = pipe_pose(prompt=prompt, negative_prompt=neg_prompt, image=pose, generator=generator, num_inference_steps=25, guidance_scale=7.5).images[0]
        img_ab2.save(os.path.join(OUTPUT_DIR, f"{scene_name}{METHOD_SUFFIX[5]}.png"))
        del img_ab2
        pipe_pose.unet.set_attn_processor(AttnProcessor())
        clear_gpu_memory()

        print(f" {scene_name} 全部生成完成！")

    print("\n 合成测试集实验完成！")
    print(f" 结果保存至：{OUTPUT_DIR}")

if __name__ == "__main__":
    run_synthetic()