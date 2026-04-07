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

# 版本校验
assert diffusers.__version__ >= "0.14.0", "请升级diffusers: pip install diffusers --upgrade"

#  工具函数 
def clear_gpu_memory():
    """显存清理"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

#  动态Token索引
def get_person_token_indices(tokenizer, prompt: str):
    """自适应 CLIP Tokenizer 拆分逻辑，匹配 person 1 和 person 2"""
    inputs = tokenizer(
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    # 转换为小写、移除 CLIP 的结尾标识符
    clean_tokens = [t.replace("</w>", "").lower() if t else "" for t in tokens]
    
    print(f"DEBUG Tokens: {clean_tokens}")

    person1_start = person2_start = None

    # 遍历 Token 序列，寻找 "person" 后面紧跟 "1" 或 "2" 的情况
    for i in range(len(clean_tokens) - 1):
        if clean_tokens[i] == "person":
            if clean_tokens[i+1] == "1":
                person1_start = i
            elif clean_tokens[i+1] == "2":
                person2_start = i

    if person1_start is None or person2_start is None:
        # 如果没找到，可能是没有空格。匹配包含关系
        for i, t in enumerate(clean_tokens):
            if "person" in t:
                if "1" in t or (i+1 < len(clean_tokens) and "1" in clean_tokens[i+1]):
                    person1_start = i
                if "2" in t or (i+1 < len(clean_tokens) and "2" in clean_tokens[i+1]):
                    person2_start = i

    if person1_start is None or person2_start is None:
        raise ValueError(f"❌ 未检测到 person1/person2。当前分词结果: {clean_tokens[:20]}...")

    # Person1 结束于 Person2 开始之前，且去掉末尾的逗号/冒号
    person1_end = person2_start - 1
    while person1_end > person1_start and clean_tokens[person1_end] in [",", ":", "."]:
        person1_end -= 1
    
    # Person2 结束 : 跳过人物内部逗号，在场景描述前截断
    person2_end = len(clean_tokens) - 1
    # 1. 找到场景描述的第一个词，根据Prompt场景词调整
    scene_start = None
    for i in range(person2_start + 2, len(clean_tokens)):
        # 匹配场景词「rural」，若换场景可修改为对应词
        if clean_tokens[i] == "rural":
            scene_start = i
            break
    if scene_start is not None:
        # 2. 找到场景词前的最后一个逗号，作为Person2的结束位置
        for i in range(scene_start - 1, person2_start, -1):
            if clean_tokens[i] in [",", "."]:
                person2_end = i - 1
                break
    else:
        # 3. 如果没找到场景词，用原来的逻辑
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

# 修复维度不匹配的注意力掩码 
class AttentionMaskProcessor:
    def __init__(self, token_indices, masks):
        self.token_indices = token_indices
        self.masks = masks

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 标准Attention计算
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * attn.scale

        # 修复维度不匹配的多人物空间掩码
        if is_cross:
            height = width = int(np.sqrt(seq_len))
            if height * width == seq_len:
                for idx, mask in enumerate(self.masks):
                    if idx >= len(self.token_indices):
                        break
                    
                    # 自适应尺寸/设备/精度
                    mask = mask.to(device=device, dtype=dtype)
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="nearest"
                    ).flatten(2)
                    
                    # 无效区域置-10000，彻底屏蔽
                    mask_neg = (1 - mask) * -10000.0
                    # ✅ 修复维度：压缩为 [1, H*W]，匹配广播需求
                    mask_neg = mask_neg.view(1, -1)
                    
                    # ✅ 直接对第三维单个token索引操作，解决维度不匹配
                    for token_idx in self.token_indices[idx]:
                        if token_idx < attn_weights.shape[-1]:
                            attn_weights[:, :, token_idx] += mask_neg

        # 原始掩码合并
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attn.head_to_batch_dim(attention_mask)
            attn_weights = attn_weights + attention_mask

        # 输出
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=attn.dropout, training=False)
        out = torch.matmul(attn_weights, value)
        out = attn.batch_to_head_dim(out)
        return attn.to_out[0](out)

def apply_attention_mask(pipeline, token_indices, masks):
    pipeline.unet.set_attn_processor(AttentionMaskProcessor(token_indices, masks))

#  掩码处理：按面积取最大2人
def process_mask(mask_path, img_size=(512,512)):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"掩码不存在: {mask_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    # 按人物面积排序，取最大2个
    person_ids = [p for p in np.unique(mask) if p > 0]
    areas = [np.sum(mask == p) for p in person_ids]
    sorted_persons = [p for _, p in sorted(zip(areas, person_ids), reverse=True)][:2]

    if len(sorted_persons) < 2:
        raise ValueError("❌ 至少需要2个人物")

    return [torch.from_numpy((mask == p).astype(np.float32)) for p in sorted_persons]

#  主实验流程 
if __name__ == "__main__":
    try:
        #  配置 
        SAMPLE_NAME = "000000069213"
        BASE_DIR = "../coco_multi_person/complete_samples_512"
        OUTPUT_DIR = "controlnet_results"
        SEED = 42      
        PROMPT = "person1: elderly asian man with natural face, black hat, glasses, dark suit, white shirt, hands behind back, person2: middle-aged asian man with natural face, gray hat, blue striped suit, purple shirt, hands in pockets, rural street, photorealistic"

        NEG_PROMPT = "cartoon, anime, ugly, blurry, low resolution, deformed, missing person, extra people, cropped, out of frame, distorted face, bad anatomy, bad hands, text, watermark, signature, disfigured, extra limbs, deformed face, asymmetrical face, blurry face, ugly face, mutated face, cross-eyed, closed eyes, open mouth, missing teeth, bad teeth, extra fingers, missing fingers, attribute mixing, clothes mixing"
       
        # 路径拼接
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pose_path = f"{BASE_DIR}/openpose/{SAMPLE_NAME}.jpg"
        depth_path = f"{BASE_DIR}/depth/{SAMPLE_NAME}.jpg"
        mask_path = f"{BASE_DIR}/mask/{SAMPLE_NAME}.png"

        # 生成器
        generator = torch.Generator("cuda").manual_seed(SEED)

        #  加载模型
        print("🔹 加载基础模型...")
        base = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16, 
            safety_checker=None
        ).to("cuda")
        base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
        token_indices = get_person_token_indices(base.tokenizer, PROMPT)

        # 加载ControlNet
        control_pose = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
        ).to("cuda")
        control_depth = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
        ).to("cuda")

        # 构建复用管道
        pipe_b2 = StableDiffusionControlNetPipeline(**base.components, controlnet=control_pose).to("cuda")
        pipe_b3 = StableDiffusionControlNetPipeline(**base.components, controlnet=[control_pose, control_depth]).to("cuda")

        # 加载控制图
        pose = Image.open(pose_path).convert("RGB")
        depth = Image.open(depth_path).convert("RGB")
        clear_gpu_memory()

        # ========== 4组对比实验 ==========
        print("\n1/4 Baseline1：纯文本生成")
        img1 = base(prompt=PROMPT, negative_prompt=NEG_PROMPT, generator=generator, num_inference_steps=50).images[0]
        img1.save(f"{OUTPUT_DIR}/{SAMPLE_NAME}_baseline1.png")
        del base, img1
        clear_gpu_memory()

        print("\n2/4 Baseline2：OpenPose约束")
        img2 = pipe_b2(prompt=PROMPT, negative_prompt=NEG_PROMPT, image=pose, controlnet_conditioning_scale=0.7, generator=generator, num_inference_steps=50).images[0]
        img2.save(f"{OUTPUT_DIR}/{SAMPLE_NAME}_baseline2.png")
        del pipe_b2, img2
        clear_gpu_memory()

        print("\n3/4 Baseline3：OpenPose+Depth约束")
        img3 = pipe_b3(prompt=PROMPT, negative_prompt=NEG_PROMPT, image=[pose, depth], controlnet_conditioning_scale=[0.7,0.5], generator=generator, num_inference_steps=50).images[0]
        img3.save(f"{OUTPUT_DIR}/{SAMPLE_NAME}_baseline3.png")
        clear_gpu_memory()

        print("\n4/4 设计方法：注意力掩码+双ControlNet")
        masks = process_mask(mask_path)
        apply_attention_mask(pipe_b3, token_indices, masks)
        img4 = pipe_b3(prompt=PROMPT, negative_prompt=NEG_PROMPT, image=[pose, depth], controlnet_conditioning_scale=[0.7,0.5], generator=generator, num_inference_steps=50).images[0]
        img4.save(f"{OUTPUT_DIR}/{SAMPLE_NAME}_method.png")
        clear_gpu_memory()

        print("\n🎉 全部实验完成！结果已保存至 controlnet_results 文件夹")

    except Exception as e:
        print(f"\n❌ 运行错误: {str(e)}")
        clear_gpu_memory()
        raise