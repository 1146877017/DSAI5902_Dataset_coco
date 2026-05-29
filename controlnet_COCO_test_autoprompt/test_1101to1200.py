import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.models.attention_processor import Attention, AttnProcessor
import cv2
import os
import gc
import diffusers
import json
import shutil

# ====================== 配置与路径 ======================
assert diffusers.__version__ >= "0.14.0", "diffusers version wrong"
SEED = 42
BASE_DIR = "../coco_multi_person/complete_samples_512"

MODES = ["original_image", "pure_background"]

# 区间控制
START_IDX = 1101  
END_IDX = 1200  

PROMPT_CONFIG = "coco_person_prompts.json"
with open(PROMPT_CONFIG, "r", encoding="utf-8") as f:
    FULL_PROMPT_LIST = json.load(f)

# 动态计算并切片数据区间
TOTAL_AVAILABLE = len(FULL_PROMPT_LIST)
actual_start = max(1, START_IDX)
actual_end = min(TOTAL_AVAILABLE, END_IDX) if END_IDX is not None else TOTAL_AVAILABLE

print(f"  全量数据集共包含 {TOTAL_AVAILABLE} 个样本。")
print(f"  已启用区间测试模式：正在提取第 {actual_start} 至第 {actual_end} 个样本（本轮共 {actual_end - actual_start + 1} 个）。")

# 转换为 Python 的 0-based 索引切片
PROMPT_LIST = FULL_PROMPT_LIST[actual_start - 1 : actual_end]

# 定义动态的指标清单文件名，防止分批运行时覆盖历史数据
MANIFEST_NAME = f"eval_manifest_{actual_start}_{actual_end}.json"

COMMON_CONFIG = {
    "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, extra fingers, missing fingers, attribute mixing, clothes mixing",
    "steps": 30,
    "cfg": 7.5,
    "pose_scale": 0.7,
    "depth_scale": 0.5
}

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def extract_sample_name(file_name):
    return os.path.splitext(file_name)[0]

# ====================== 功能模块 ======================
def get_person_token_indices(tokenizer, prompt: str):
    """
    从右至左反向查找 Token
    """
    inputs = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    person1_start = person2_start = None

    # 反向扫描，精确捕捉句尾固定控制标识符
    for i in range(len(clean_tokens) - 2, -1, -1):
        if clean_tokens[i] == "person":
            if clean_tokens[i+1] == "1" and person1_start is None:
                person1_start = i
            elif clean_tokens[i+1] == "2" and person2_start is None:
                person2_start = i
        if person1_start is not None and person2_start is not None:
            break

    if person1_start is None or person2_start is None:
        raise ValueError(f" 未在 Prompt 中检测到标准的 person 1 / person 2 结构: {prompt}")

    person1_end = person2_start - 1
    while person1_end > person1_start and clean_tokens[person1_end] in [",", ":", "."]:
        person1_end -= 1
    
    person2_end = len(clean_tokens) - 1
    for i in range(person2_start + 2, len(clean_tokens)):
        if clean_tokens[i] in ["<|endoftext|>", ""]:
            person2_end = i - 1
            break

    token_indices = [
        list(range(person1_start, person1_end + 1)),
        list(range(person2_start, person2_end + 1))
    ]
    return token_indices

class AttentionMaskProcessor:
    """空间解耦跨注意力掩码处理器"""
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
                total_heads_in_batch = attn_weights.shape[0]
                num_heads = attn.heads  # 定义单头数量，用于在第一维(2*num_heads)中精准切片出正向条件通道
                cond_start_idx = total_heads_in_batch - num_heads if total_heads_in_batch > num_heads else 0
                
                for idx, mask in enumerate(self.masks):
                    if idx >= len(self.token_indices):
                        break
                    
                    mask = mask.to(device=device, dtype=dtype)
                    # 用双线性插值进行下采样，防止小目标在低分辨率层（如 8x8）下直接丢失
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
                    ).flatten(2)
                    
                    mask_neg = (1 - mask) * -10000.0
                    mask_neg = mask_neg.view(1, -1)
                    
                    for token_idx in self.token_indices[idx]:
                        if token_idx < attn_weights.shape[-1]:
                            # 运用动态索引
                            attn_weights[cond_start_idx:, :, token_idx] += mask_neg

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
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码不存在: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    person_ids = [p for p in np.unique(mask) if p > 0]
    
    centroids_x = []
    valid_persons = []
    for p in person_ids:
        instance_mask = (mask == p).astype(np.uint8)
        M = cv2.moments(instance_mask)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            centroids_x.append(cX)
            valid_persons.append(p)

    if len(valid_persons) < 2:
        raise ValueError(f" 错误: 样本 {mask_path} 包含的有效实例少于 2 人。")

    sorted_persons = [p for _, p in sorted(zip(centroids_x, valid_persons))][:2]
    return [torch.from_numpy((mask == p).astype(np.float32)) for p in sorted_persons]

# ====================== 模型单次全量加载 ======================
def load_all_models():
    print("  正在统一加载并初始化所有基础与控制模型...")
    base = StableDiffusionPipeline.from_pretrained(
        "Lykon/AbsoluteReality", 
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to("cuda")
    base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
    
    control_pose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
    ).to("cuda")
    control_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    ).to("cuda")
    
    pipe_b2 = StableDiffusionControlNetPipeline(**base.components, controlnet=control_pose).to("cuda")
    pipe_b3 = StableDiffusionControlNetPipeline(**base.components, controlnet=[control_pose, control_depth]).to("cuda")
    
    return base, pipe_b2, pipe_b3

# ====================== 核心控制实验单步运行 ======================
def run_single_sample(prompt_item, base, pipe_b2, pipe_b3, generator, mode, is_first_mode, global_idx):
    """完整跑完单个COCO样本实验"""
    file_name = prompt_item["file_name"]
    sample_name = extract_sample_name(file_name)
    prompt = prompt_item["prompt"]
    img_id = prompt_item["image_id"]

    neg_prompt = COMMON_CONFIG["NEG_PROMPT"]
    parent_steps = COMMON_CONFIG["steps"]
    cfg = COMMON_CONFIG["cfg"]
    pose_scale = COMMON_CONFIG["pose_scale"]
    depth_scale = COMMON_CONFIG["depth_scale"]

    output_dir = f"results_{START_IDX}to{END_IDX}_{mode}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}\n  [{mode.upper()}] 正在处理核心样本 [全局第 {global_idx} 个]: {sample_name}\n{'='*60}")
    
    pose_path = f"{BASE_DIR}/openpose/{sample_name}.png"
    depth_path = f"{BASE_DIR}/depth_{mode}/{sample_name}.png"  
    mask_path = f"{BASE_DIR}/mask/{sample_name}.png"

    for path in [pose_path, depth_path, mask_path]:
        if not os.path.exists(path):
            print(f"  必要控制图缺失，跳过该样本: {path}")
            return None

    try:
        pose = Image.open(pose_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        token_indices = get_person_token_indices(base.tokenizer, prompt)
    except Exception as e:
        print(f"  前置数据分析失败: {str(e)}")
        return None

    # --- 1. Baseline1：纯文本生成 ---
    b1_path = f"{output_dir}/{sample_name}_baseline1.png"
    if is_first_mode:
        try:
            print("  [1/4] 正在运行 Baseline1：纯文本生成...")
            generator.manual_seed(SEED)  
            img1 = base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
            img1.save(b1_path)
            del img1
            clear_gpu_memory()
        except Exception as e: print(f"   Baseline1 失败: {e}")
    else:                
        src_b1 = f"results_{START_IDX}to{END_IDX}_{MODES[0]}/{sample_name}_baseline1.png"
        if os.path.exists(src_b1):
            shutil.copy(src_b1, b1_path)
        else:
            print(f"  未找到历史 Baseline1 缓存，正在现场重新启动渲染...")
            try:
                generator.manual_seed(SEED)  
                img1 = base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
                img1.save(b1_path)
                del img1
                clear_gpu_memory()
            except Exception as e: print(f"   现场补算 Baseline1 失败: {e}")

    # --- 2. Baseline2：仅 OpenPose ---
    b2_path = f"{output_dir}/{sample_name}_baseline2.png"
    if is_first_mode:
        try:
            print("  [2/4] 正在运行 Baseline2：单 OpenPose 引导...")
            generator.manual_seed(SEED)  
            img2 = pipe_b2(prompt=prompt, negative_prompt=neg_prompt, image=pose, controlnet_conditioning_scale=pose_scale, generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
            img2.save(b2_path)
            del img2
            clear_gpu_memory()
        except Exception as e: print(f"   Baseline2 失败: {e}")
    else:        
        src_b2 = f"results_{START_IDX}to{END_IDX}_{MODES[0]}/{sample_name}_baseline2.png"
        if os.path.exists(src_b2):
            shutil.copy(src_b2, b2_path)
        else:
            print(f"  未找到历史 Baseline2 缓存，正在现场重新启动航向引导...")
            try:
                generator.manual_seed(SEED)  
                img2 = pipe_b2(prompt=prompt, negative_prompt=neg_prompt, image=pose, controlnet_conditioning_scale=pose_scale, generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
                img2.save(b2_path)
                del img2
                clear_gpu_memory()
            except Exception as e: print(f"   现场补算 Baseline2 失败: {e}")

    # --- 3. Baseline3：双 ControlNet ---
    try:
        print(f"  [3/4] 正在运行 Baseline3：双 ControlNet 强组合 ({mode})...")
        generator.manual_seed(SEED)  
        img3 = pipe_b3(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], controlnet_conditioning_scale=[pose_scale, depth_scale], generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
        img3.save(f"{output_dir}/{sample_name}_baseline3.png")
        del img3
        clear_gpu_memory()
    except Exception as e: print(f"   Baseline3 失败: {e}")
    
    # --- 4. Method：空间解耦跨注意力掩码 + 双 ControlNet ---
    # 完整备份当前的加速注意力处理器字典
    orig_processors = pipe_b3.unet.attn_processors
    try:
        print(f"  [4/4] 正在运行创新方法：跨注意力掩码 + 双 ControlNet 联合干预 ({mode})...")
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_b3, token_indices, masks)
        
        generator.manual_seed(SEED)  
        img4 = pipe_b3(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], controlnet_conditioning_scale=[pose_scale, depth_scale], generator=generator, num_inference_steps=parent_steps, guidance_scale=cfg).images[0]
        img4.save(f"{output_dir}/{sample_name}_method.png")
        del img4
        clear_gpu_memory()
    except Exception as e: 
        print(f"   Method 方法运行失败: {str(e)}")
    finally:
        # 原样恢复原有的加速状态
        pipe_b3.unet.set_attn_processor(orig_processors)

    return {
        "image_id": img_id,
        "sample_name": sample_name,
        "mode": mode,
        "method_img_path": f"{output_dir}/{sample_name}_method.png",
        "person1_gt": prompt_item["person1_gt"],
        "person2_gt": prompt_item["person2_gt"]
    }

# ====================== 主运行循环 ======================
if __name__ == "__main__":
    try:
        generator = torch.Generator("cuda").manual_seed(SEED)
        base, pipe_b2, pipe_b3 = load_all_models()
        
        round_samples = len(PROMPT_LIST)
        print(f"\n  数据集切片就绪，全面启动批处理，本轮测试样本数: {round_samples}")
        
        eval_manifest = []
        
        for mode_idx, mode in enumerate(MODES):
            print(f"\n{'*'*40}\n  开始执行模式实验: results_{mode} \n{'*'*40}")
            is_first_mode = (mode_idx == 0)
            
            for idx, prompt_item in enumerate(PROMPT_LIST):
                # 计算全局数据集中的真实序号（1-based）
                global_idx = actual_start + idx
                print(f" 当前模式 [{mode}] 进度: [ {idx+1} / {round_samples} ] (数据集总第 {global_idx} 个)")
                
                record = run_single_sample(prompt_item, base, pipe_b2, pipe_b3, generator, mode, is_first_mode, global_idx)
                if record:
                    eval_manifest.append(record)

        # 动态将清单持久化到对应的区间文件中
        with open(MANIFEST_NAME, "w", encoding="utf-8") as f:
            json.dump(eval_manifest, f, indent=4, ensure_ascii=False)
        print(f"\n  已成功自动更新本轮实验评估清单: `{MANIFEST_NAME}`")
        

    except Exception as e:
        print(f"\n 顶层中断错误: {str(e)}")
        clear_gpu_memory()
        raise