import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionPipeline
)
from diffusers.models.attention_processor import Attention
import cv2
import os
import gc
import diffusers

# ====================== 全局配置 ======================
assert diffusers.__version__ >= "0.14.0", "diffusers version wrong"
SEED = 42
BASE_DIR = "../coco_multi_person/complete_samples_512"
OUTPUT_DIR = "controlnet_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 工具函数 ======================
def clear_gpu_memory():
    """gpu"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ====================== 功能模块 ======================
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

    # 兼容无空格格式（person1/person2）
    if person1_start is None or person2_start is None:
        for i, t in enumerate(clean_tokens):
            if "person" in t:
                if "1" in t:
                    person1_start = i
                if "2" in t:
                    person2_start = i

    if person1_start is None or person2_start is None:
        raise ValueError(f"❌ 未检测到 person1/person2")

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
    
    print(f"✅ Token 匹配成功")
    print(f"   Person1 indices: {token_indices[0]} -> {[clean_tokens[i] for i in token_indices[0]]}")
    print(f"   Person2 indices: {token_indices[1]} -> {[clean_tokens[i] for i in token_indices[1]]}")
    
    return token_indices

class AttentionMaskProcessor:
    """注意力掩码核心类"""
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
    """掩码处理：提取最大2个人物区域"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码不存在: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    person_ids = [p for p in np.unique(mask) if p > 0]
    areas = [np.sum(mask == p) for p in person_ids]
    sorted_persons = [p for _, p in sorted(zip(areas, person_ids), reverse=True)][:2]

    if len(sorted_persons) < 2:
        raise ValueError("❌ 至少需要2个人物")

    return [torch.from_numpy((mask == p).astype(np.float32)) for p in sorted_persons]

# ====================== 样本配置 ======================
SAMPLE_CONFIGS = [
    # 1. 000000069213
    {
        "SAMPLE_NAME": "000000069213",
        "PROMPT": "person1: elderly asian man with natural face, black hat, glasses, dark suit, white shirt, hands behind back, person2: middle-aged asian man with natural face, gray hat, blue striped suit, purple shirt, hands in pockets, rural street, photorealistic",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, extra fingers, missing fingers, attribute mixing, clothes mixing",
        "steps": 50,
        "cfg": 7.5,
        "pose_scale": 0.7,
        "depth_scale": 0.5
    },
    # 2. 000000001268
    {
        "SAMPLE_NAME": "000000001268",
        "PROMPT": "person1: a man wearing a red jacket, person2: a woman wearing a blue dress, realistic, high detail, city street background",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed",
        "steps": 30,
        "cfg": 7.5,
        "pose_scale": 0.7,
        "depth_scale": 0.5
    },
    # 3. 000000008690
    {
        "SAMPLE_NAME": "000000008690",
        "PROMPT": "person1: (a young asian girl wearing a pink floral dress with a pink bow, petting a black goat:1.2), person2: (a young asian girl wearing a blue dress with pink straps and strawberry prints, smiling at the camera:1.2), outdoor petting zoo, green grass, white metal fence, sunny day, photorealistic, high detail, 4k",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs",
        "steps": 30,
        "cfg": 7.5,
        "pose_scale": 0.7,
        "depth_scale": 0.5
    },
    # 4. 000000492362
    {
        "SAMPLE_NAME": "000000492362",
        "PROMPT": "person1: young man in red t-shirt, blue jeans, red boots, standing on skateboard, holding phone, person2: woman in floral white top, black pants, holding orange bag, street food cart, photorealistic",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, extra fingers, missing fingers, attribute mixing, clothes mixing",
        "steps": 50,
        "cfg": 7.5,
        "pose_scale": 0.7,
        "depth_scale": 0.5
    },
    # 5. 000000504074
    {
        "SAMPLE_NAME": "000000504074",
        "PROMPT": "person1: young woman with glasses, tank top, using laptop, smoking, sitting in chair, person2: young woman with sunglasses, white top, holding wine glass, sitting in chair, rooftop patio, black and white photo, photorealistic",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, extra fingers, missing fingers, attribute mixing, clothes mixing, color photo, colored",
        "steps": 50,
        "cfg": 7.5,
        "pose_scale": 0.7,
        "depth_scale": 0.5
    },
    # 6. 000000044279
    {
        "SAMPLE_NAME": "000000044279",
        "PROMPT": "person1: (a middle-aged asian male chef with natural face, detailed facial features, white shirt packing takeout food:1.3), person2: (a middle-aged asian male chef with natural face, detailed facial features, white shirt blue cap cooking at stove:1.3), chinese restaurant kitchen, photorealistic, high detail",
        "NEG_PROMPT": "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth",
        "steps": 50,
        "cfg": 8.5,
        "pose_scale": 0.65,
        "depth_scale": 0.45
    }
]

# ====================== 统一模型加载 ======================
def load_all_models():
    print("🔹 统一加载所有模型...")
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

# ====================== 实验流程 ======================
def run_single_sample(config, base, pipe_b2, pipe_b3, generator):
    sample_name = config["SAMPLE_NAME"]
    prompt = config["PROMPT"]
    neg_prompt = config["NEG_PROMPT"]
    steps = config["steps"]
    cfg = config["cfg"]
    pose_scale = config["pose_scale"]
    depth_scale = config["depth_scale"]

    print(f"\n{'='*50}\n 开始处理样本: {sample_name}\n{'='*50}")
    
    # 路径
    pose_path = f"{BASE_DIR}/openpose/{sample_name}.jpg"
    depth_path = f"{BASE_DIR}/depth/{sample_name}.jpg"
    mask_path = f"{BASE_DIR}/mask/{sample_name}.png"
    
    # 加载控制图
    pose = Image.open(pose_path).convert("RGB")
    depth = Image.open(depth_path).convert("RGB")
    token_indices = get_person_token_indices(base.tokenizer, prompt)

    # 1. Baseline1：纯文本
    print("\n1/4 Baseline1：纯文本生成")
    img1 = base(prompt=prompt, negative_prompt=neg_prompt, generator=generator, num_inference_steps=steps, guidance_scale=cfg).images[0]
    img1.save(f"{OUTPUT_DIR}/{sample_name}_baseline1.png")
    del img1
    clear_gpu_memory()

    # 2. Baseline2：OpenPose
    print("\n2/4 Baseline2：OpenPose")
    img2 = pipe_b2(prompt=prompt, negative_prompt=neg_prompt, image=pose, controlnet_conditioning_scale=pose_scale, generator=generator, num_inference_steps=steps, guidance_scale=cfg).images[0]
    img2.save(f"{OUTPUT_DIR}/{sample_name}_baseline2.png")
    del img2
    clear_gpu_memory()

    # 3. Baseline3：双ControlNet
    print("\n3/4 Baseline3：OpenPose+Depth")
    img3 = pipe_b3(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], controlnet_conditioning_scale=[pose_scale, depth_scale], generator=generator, num_inference_steps=steps, guidance_scale=cfg).images[0]
    img3.save(f"{OUTPUT_DIR}/{sample_name}_baseline3.png")
    del img3
    clear_gpu_memory()

    # 4. 方法：注意力掩码+双ControlNet
    print("\n4/4 注意力掩码+双ControlNet")
    masks = process_mask(mask_path)
    apply_attention_mask(pipe_b3, token_indices, masks)
    img4 = pipe_b3(prompt=prompt, negative_prompt=neg_prompt, image=[pose, depth], controlnet_conditioning_scale=[pose_scale, depth_scale], generator=generator, num_inference_steps=steps, guidance_scale=cfg).images[0]
    img4.save(f"{OUTPUT_DIR}/{sample_name}_method.png")
    del img4
    # 重置
    pipe_b3.unet.set_attn_processor(AttnProcessor())
    clear_gpu_memory()

    print(f"\n 样本 {sample_name} 处理完成")

# ====================== 主函数 ======================
if __name__ == "__main__":
    try:
        # 初始化
        generator = torch.Generator("cuda").manual_seed(SEED)
        base, pipe_b2, pipe_b3 = load_all_models()
        # 导入默认注意力处理器
        from diffusers.models.attention_processor import AttnProcessor
        
        # 循环处理所有样本
        for config in SAMPLE_CONFIGS:
            run_single_sample(config, base, pipe_b2, pipe_b3, generator)

        print("\n✅ 所有样本全部处理完成！结果保存在 controlnet_results 文件夹")

    except Exception as e:
        print(f"\n❌ 运行错误: {str(e)}")
        clear_gpu_memory()
        raise