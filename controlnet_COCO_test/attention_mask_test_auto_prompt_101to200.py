import sys
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.models.attention_processor import Attention, AttnProcessor  # 导入AttnProcessor
from gfpgan import GFPGANer
import cv2
import os
import gc
import diffusers

import json
PROMPT_CONFIG = "coco_person_prompts.json"
with open(PROMPT_CONFIG, "r", encoding="utf-8") as f:
    PROMPT_LIST = json.load(f)

# ====================== 配置 ======================
assert diffusers.__version__ >= "0.14.0", "diffusers version wrong"
SEED = 42
BASE_DIR = "../coco_multi_person/complete_samples_512"
OUTPUT_DIR = "controlnet_results_101to200"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 通用生成参数
COMMON_CONFIG = {
    "NEG_PROMPT": (
        "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, "
        # 人脸修复负面词
        "bad eyes, bad nose, bad mouth, deformed lips, asymmetric eyes, ugly chin, "
        "mangled face, fused face, cloned face, poorly drawn face, morbid, "
        "extra fingers, missing fingers, attribute mixing, clothes mixing, "
    ),
    "steps": 35,
    "cfg": 7.0,
    "pose_scale": 0.5,
    "depth_scale": 0.3
}

# 初始化人脸修复器
face_restorer = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=1,  # 不放大图片，只修复人脸
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)


# ====================== 工具函数 ======================
def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def extract_sample_name(file_name):
    """从file_name提取样本ID"""
    return os.path.splitext(file_name)[0]

# ====================== 功能模块 ======================
def fix_face(image_path):
    """修复图片里的所有人脸"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    _, _, restored_img = face_restorer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )
    cv2.imwrite(image_path, restored_img)

def get_person_token_indices(tokenizer, prompt: str):
    """拆分，匹配 person 1 和 person 2"""
    inputs = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    person1_start = person2_start = None

    # 匹配 person + 1/2
    for i in range(len(clean_tokens) - 1):
        if clean_tokens[i] == "person":
            if clean_tokens[i+1] == "1":
                person1_start = i
            elif clean_tokens[i+1] == "2":
                person2_start = i

    # 兼容无空格格式
    if person1_start is None or person2_start is None:
        for i, t in enumerate(clean_tokens):
            if "person" in t:
                if "1" in t:
                    person1_start = i
                if "2" in t:
                    person2_start = i

    if person1_start is None or person2_start is None:
        raise ValueError(f" 未检测到 person1/person2 in prompt: {prompt}")

    # 截断Token区间
    person1_end = person2_start - 1
    while person1_end > person1_start and clean_tokens[person1_end] in [",", ":", "."]:
        person1_end -= 1
    
    person2_end = len(clean_tokens) - 1
    for i in range(person2_start + 2, len(clean_tokens)):
        if clean_tokens[i] in [",", ".", "<|endoftext|>"]:
            person2_end = i - 1
            break

    token_indices = [
        list(range(person1_start, person1_end + 1)),
        list(range(person2_start, person2_end + 1))
    ]
    
    print(f" Token 匹配成功")
    print(f"   Person1 indices: {token_indices[0]} -> {[clean_tokens[i] for i in token_indices[0]]}")
    print(f"   Person2 indices: {token_indices[1]} -> {[clean_tokens[i] for i in token_indices[1]]}")
    
    return token_indices

class AttentionMaskProcessor:
    """注意力掩码类"""
    def __init__(self, token_indices, masks):
        self.token_indices = token_indices
        self.masks = masks

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        device = hidden_states.device
        dtype = hidden_states.dtype

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

        if is_cross:
            height = width = int(np.sqrt(seq_len))
            if height * width == seq_len:
                for idx, mask in enumerate(self.masks):
                    if idx >= len(self.token_indices):
                        break
                    
                    mask = mask.to(device=device, dtype=dtype)
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="nearest"
                    ).flatten(2)
                    
                    mask_neg = (1 - mask) * -10000.0
                    mask_neg = mask_neg.view(1, -1)
                    
                    for token_idx in self.token_indices[idx]:
                        if token_idx < attn_weights.shape[-1]:
                            attn_weights[:, :, token_idx] += mask_neg

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attn.head_to_batch_dim(attention_mask)
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=attn.dropout, training=False)
        out = torch.matmul(attn_weights, value)
        out = attn.batch_to_head_dim(out)
        return attn.to_out[0](out)

def apply_attention_mask(pipeline, token_indices, masks):
    pipeline.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks))

def process_mask(mask_path, img_size=(512,512)):
    """提取最大2个人物区域"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码不存在: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    person_ids = [p for p in np.unique(mask) if p > 0]
    areas = [np.sum(mask == p) for p in person_ids]
    sorted_persons = [p for _, p in sorted(zip(areas, person_ids), reverse=True)][:2]

    if len(sorted_persons) < 2:
        raise ValueError("Err : 至少需要2个人物")

    return [torch.from_numpy((mask == p).astype(np.float32)) for p in sorted_persons]

# ====================== 模型加载 ======================
def load_all_models():
    print(" 统一加载所有模型...")
    # 基础模型
    base = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
    
    # ControlNet模型
    control_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
    ).to("cuda")
    control_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    ).to("cuda")
    
    # 构建管道
    pipe_b2 = StableDiffusionControlNetPipeline(**base.components, controlnet=control_pose).to("cuda")
    pipe_b3 = StableDiffusionControlNetPipeline(**base.components, controlnet=[control_pose, control_depth]).to("cuda")
    
    return base, pipe_b2, pipe_b3

# ====================== 实验 ======================
def run_single_sample(prompt_item, base, pipe_b2, pipe_b3, generator):
    """处理单个COCO样本"""
    # 从prompt_item提取核心信息
    file_name = prompt_item["file_name"]
    sample_name = extract_sample_name(file_name)  # 提取ID
    prompt = prompt_item["prompt"]
    img_id = prompt_item["image_id"]

    # 通用参数
    neg_prompt = COMMON_CONFIG["NEG_PROMPT"]
    steps = COMMON_CONFIG["steps"]
    cfg = COMMON_CONFIG["cfg"]
    pose_scale = COMMON_CONFIG["pose_scale"]
    depth_scale = COMMON_CONFIG["depth_scale"]

    print(f"\n{'='*50}\n 开始处理样本: {sample_name} (img_id: {img_id})\n{'='*50}")
    
    # 路径
    pose_path = f"{BASE_DIR}/openpose/{sample_name}.jpg"
    depth_path = f"{BASE_DIR}/depth/{sample_name}.jpg"
    mask_path = f"{BASE_DIR}/mask/{sample_name}.png"

    # 检查文件是否存在
    for path in [pose_path, depth_path, mask_path]:
        if not os.path.exists(path):
            print(f"  文件不存在，跳过样本 {sample_name}: {path}")
            return

    # 加载控制图
    try:
        pose = Image.open(pose_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        token_indices = get_person_token_indices(base.tokenizer, prompt)
    except Exception as e:
        print(f"  加载/解析失败，跳过样本 {sample_name}: {str(e)}")
        return

    # 1. Baseline1：纯文本
    try:
        print("\n1/4 Baseline1：纯文本生成")
        img1 = base(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            generator=generator, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
        img1.save(f"{OUTPUT_DIR}/{sample_name}_baseline1.png")
        fix_face(f"{OUTPUT_DIR}/{sample_name}_baseline1.png")
        del img1
        clear_gpu_memory()
    except Exception as e:
        print(f" Baseline1 失败: {str(e)}")

    # 2. Baseline2：OpenPose
    try:
        print("\n2/4 Baseline2：OpenPose")
        img2 = pipe_b2(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=pose, 
            controlnet_conditioning_scale=pose_scale, 
            generator=generator, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
        img2.save(f"{OUTPUT_DIR}/{sample_name}_baseline2.png")
        fix_face(f"{OUTPUT_DIR}/{sample_name}_baseline2.png")
        del img2
        clear_gpu_memory()
    except Exception as e:
        print(f" Baseline2 失败: {str(e)}")

    # 3. Baseline3：双ControlNet
    try:
        print("\n3/4 Baseline3：OpenPose+Depth")
        img3 = pipe_b3(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=[pose, depth], 
            controlnet_conditioning_scale=[pose_scale, depth_scale], 
            generator=generator, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
        img3.save(f"{OUTPUT_DIR}/{sample_name}_baseline3.png")
        fix_face(f"{OUTPUT_DIR}/{sample_name}_baseline3.png")
        del img3
        clear_gpu_memory()
    except Exception as e:
        print(f" Baseline3 失败: {str(e)}")

    # 4. 方法：注意力掩码+双ControlNet
    try:
        print("\n4/4 注意力掩码+双ControlNet")
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_b3, token_indices, masks)
        img4 = pipe_b3(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=[pose, depth], 
            controlnet_conditioning_scale=[pose_scale, depth_scale], 
            generator=generator, 
            num_inference_steps=steps, 
            guidance_scale=cfg
        ).images[0]
        img4.save(f"{OUTPUT_DIR}/{sample_name}_method.png")
        fix_face(f"{OUTPUT_DIR}/{sample_name}_method.png")
        del img4
        clear_gpu_memory()
    except Exception as e:
        print(f" 注意力掩码方法失败: {str(e)}")
    finally:
        # 重置注意力处理器
        pipe_b3.unet.set_attn_processor(AttnProcessor())

    print(f"\n 样本 {sample_name} 处理完成")

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 初始化
        generator = torch.Generator("cuda").manual_seed(SEED)
        base, pipe_b2, pipe_b3 = load_all_models()
        
        # 循环处理所有COCO样本
        total_samples = len(PROMPT_LIST)
        # 取前100个
        process_samples = PROMPT_LIST[100:200]
        print(f"\n 开始批量处理 {len(process_samples)} （101到200）个COCO样本")
        for idx, prompt_item in enumerate(process_samples):
            print(f"\n--- 进度: {idx+1}/{len(process_samples)} ---")
            run_single_sample(prompt_item, base, pipe_b2, pipe_b3, generator)

        print(f"\n 所有样本处理完成！结果保存在 {OUTPUT_DIR} 文件夹")

    except Exception as e:
        print(f"\n 运行错误: {str(e)}")
        clear_gpu_memory()
        raise