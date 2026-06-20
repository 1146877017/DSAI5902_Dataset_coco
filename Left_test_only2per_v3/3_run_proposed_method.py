import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from config import *

def load_condition_and_masks(condition_dir):
    """加载控制图与角色掩码"""
    pose = Image.open(os.path.join(condition_dir, "pose.png")).convert("RGB")
    depth = Image.open(os.path.join(condition_dir, "depth.png")).convert("RGB")
    mask_a = cv2.imread(os.path.join(condition_dir, "mask_a.png"), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(os.path.join(condition_dir, "mask_b.png"), cv2.IMREAD_GRAYSCALE)
    return pose, depth, mask_a, mask_b

def build_prompt_and_token_ranges(pipe, char_a_name, char_b_name, background):
    """构建提示词并计算两个角色描述对应的token索引范围"""
    char_a = CHARACTER_CONFIGS[char_a_name]
    char_b = CHARACTER_CONFIGS[char_b_name]

    desc_a = f"{char_a['trigger']}, {char_a['desc']}"
    desc_b = f"{char_b['trigger']}, {char_b['desc']}"
    full_prompt = (
        f"left character: {desc_a}, right character: {desc_b}, "
        f"{background} background, full body, masterpiece, best quality, anime style"
    )

    # 计算token范围
    tokenizer = pipe.tokenizer
    full_tokens = tokenizer.encode(full_prompt)
    tokens_a = tokenizer.encode(desc_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(desc_b, add_special_tokens=False)

    # 匹配角色描述在完整提示词中的token起始位置
    start_a = None
    for i in range(len(full_tokens)):
        if full_tokens[i:i+len(tokens_a)] == tokens_a:
            start_a = i
            break
    range_a = (start_a, start_a + len(tokens_a))

    start_b = None
    for i in range(len(full_tokens)):
        if full_tokens[i:i+len(tokens_b)] == tokens_b:
            start_b = i
            break
    range_b = (start_b, start_b + len(tokens_b))

    return full_prompt, range_a, range_b

def register_attention_mask_hooks(pipe, token_range_a, token_range_b, mask_a, mask_b):
    """注册交叉注意力掩码钩子，将角色token的注意力限制在对应空间区域"""
    # 掩码下采样到 latent 分辨率（原图 1/8）
    h_latent = IMAGE_HEIGHT // 8
    w_latent = IMAGE_WIDTH // 8

    mask_a_down = cv2.resize(mask_a, (w_latent, h_latent)) / 255.0
    mask_b_down = cv2.resize(mask_b, (w_latent, h_latent)) / 255.0
    mask_a_tensor = torch.from_numpy(mask_a_down).float().to("cuda").flatten()
    mask_b_tensor = torch.from_numpy(mask_b_down).float().to("cuda").flatten()

    def attention_mask_hook(module, input, output):
        # output[1] 为注意力权重矩阵 shape: [batch*heads, seq_len, seq_len]
        attn_weights = output[1]
        seq_len = attn_weights.shape[2]

        # 对角色A的token范围应用空间掩码
        s_a, e_a = token_range_a
        if e_a <= seq_len:
            attn_weights[:, s_a:e_a, :] *= mask_a_tensor.unsqueeze(0).unsqueeze(1)

        # 对角色B的token范围应用空间掩码
        s_b, e_b = token_range_b
        if e_b <= seq_len:
            attn_weights[:, s_b:e_b, :] *= mask_b_tensor.unsqueeze(0).unsqueeze(1)

        return (output[0], attn_weights)

    # 注册到所有交叉注意力层（attn2）
    hooks = []
    for name, module in pipe.unet.named_modules():
        if "attn2" in name and "to_k" not in name and "to_v" not in name:
            hook = module.register_forward_hook(attention_mask_hook)
            hooks.append(hook)
    return hooks

def init_proposed_pipeline():
    """初始化所提方法的管线（双ControlNet）"""
    controlnets = [
        ControlNetModel.from_pretrained(CONTROLNET_OPENPOSE, torch_dtype=torch.float16),
        ControlNetModel.from_pretrained(CONTROLNET_DEPTH, torch_dtype=torch.float16)
    ]
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        BASE_MODEL_PATH,
        controlnet=controlnets,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

def load_dual_lora(pipe, char_a_name, char_b_name):
    char_a = CHARACTER_CONFIGS[char_a_name]
    char_b = CHARACTER_CONFIGS[char_b_name]
    pipe.load_lora_weights(os.path.join(LORA_DIR, char_a["file"]), adapter_name="char_a")
    pipe.load_lora_weights(os.path.join(LORA_DIR, char_b["file"]), adapter_name="char_b")
    pipe.set_adapters(["char_a", "char_b"], adapter_weights=[LORA_WEIGHT, LORA_WEIGHT])
    return pipe

def main():
    print("===== Running Proposed Method =====")
    pipe = init_proposed_pipeline()
    generator = torch.Generator("cuda").manual_seed(GLOBAL_SEED)

    for scene in SCENES:
        for bg in BACKGROUNDS:
            for (char_a, char_b) in CHARACTER_PAIRS:
                condition_dir = os.path.join(CONDITION_DIR, scene, bg, f"{char_a}_{char_b}")
                save_dir = os.path.join(OUTPUT_ROOT, "proposed", scene, bg, f"{char_a}_{char_b}")
                os.makedirs(save_dir, exist_ok=True)

                # 加载条件与掩码
                pose_img, depth_img, mask_a, mask_b = load_condition_and_masks(condition_dir)

                # 加载LoRA、构建提示词与token范围
                pipe = load_dual_lora(pipe, char_a, char_b)
                prompt, range_a, range_b = build_prompt_and_token_ranges(pipe, char_a, char_b, bg)

                # 注册注意力掩码钩子
                hooks = register_attention_mask_hooks(pipe, range_a, range_b, mask_a, mask_b)

                # 生成图像
                image = pipe(
                    prompt=prompt,
                    image=[pose_img, depth_img],
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=CFG_SCALE,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    generator=generator
                ).images[0]

                # 移除钩子，避免影响后续样本
                for hook in hooks:
                    hook.remove()

                image.save(os.path.join(save_dir, "result.png"))
                print(f"Saved: {save_dir}/result.png")

                pipe.unload_lora_weights()

    del pipe
    torch.cuda.empty_cache()
    print("Proposed method completed.")

if __name__ == "__main__":
    main()