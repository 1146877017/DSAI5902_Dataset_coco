import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import Attention
import cv2
import os

# 修改 Attention 层，加入 Attention Masking 
def apply_attention_masking(
    pipeline, 
    attention_masks: list[torch.Tensor],
    token_indices: list[list[int]]
):
    downsample_rate = 8

    def custom_cross_attention_forward(
        self: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if encoder_hidden_states is not None and len(attention_masks) > 0:
            attn_mask = torch.zeros_like(attention_mask) if attention_mask is not None else None
            for person_idx, mask in enumerate(attention_masks):
                for token_idx in token_indices[person_idx]:
                    if token_idx >= sequence_length:
                        continue
                    mask_upscaled = mask.unsqueeze(0).unsqueeze(0)
                    mask_flat = mask_upscaled.reshape(1, 1, -1)
                    if attn_mask is None:
                        attn_mask = mask_flat
                    else:
                        attn_mask[:, token_idx:token_idx+1, :] = mask_flat
        
        return self._original_forward(
            hidden_states, encoder_hidden_states, attention_mask=attn_mask, **kwargs
        )

    for name, module in pipeline.unet.named_modules():
        if isinstance(module, Attention) and "cross" in name:
            module._original_forward = module.forward
            module.forward = custom_cross_attention_forward.__get__(module, Attention)

# 处理实例级 Mask 为注意力掩码 
def process_instance_mask(mask_path: str, target_size=(512, 512)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    person_ids = np.unique(mask)
    person_ids = [pid for pid in person_ids if pid > 0]
    attention_masks = []
    
    for pid in person_ids:
        person_mask = (mask == pid).astype(np.float32)
        h, w = person_mask.shape
        person_mask_down = cv2.resize(
            person_mask, (w//8, h//8), interpolation=cv2.INTER_NEAREST
        )
        attention_masks.append(torch.from_numpy(person_mask_down).to("cuda"))
    
    return attention_masks

# 加载模型 + 推理 
if __name__ == "__main__":
    # 1. 使用统一尺寸后的 512×512 数据集
    sample_name = "000000001268"
    openpose_path = f"coco_multi_person/complete_samples_512/openpose/{sample_name}.jpg"
    depth_path = f"coco_multi_person/complete_samples_512/depth/{sample_name}.jpg"
    mask_path = f"coco_multi_person/complete_samples_512/mask/{sample_name}.png"
    
    output_dir = "controlnet_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载 ControlNet 模型
    controlnet_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to("cuda")
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16
    ).to("cuda")

    # 3. 加载 Stable Diffusion 管道
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[controlnet_openpose, controlnet_depth],
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_xformers_memory_efficient_attention()

    # 4. 已经调整为512×512，无需resize
    openpose_img = Image.open(openpose_path).convert("RGB")
    depth_img = Image.open(depth_path).convert("RGB")

    # 5. 提示词
    prompt = "person1: a man wearing a red jacket, person2: a woman wearing a blue dress, realistic, high detail, city street background"
    negative_prompt = "cartoon, anime, ugly, blurry, low resolution, deformed"
    
    # 6. token 索引
    tokenizer = pipe.tokenizer
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    token_indices = [
        [2,3,4,5],
        [7,8,9,10]
    ]

    #  实验1：基线 
    print("生成基线结果...")
    image_baseline = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[openpose_img, depth_img],
        controlnet_conditioning_scale=[0.7, 0.5],
        num_inference_steps=30,
        generator=torch.manual_seed(42),
        width=512,
        height=512
    ).images[0]
    image_baseline.save(f"{output_dir}/{sample_name}_baseline.png")

    #  实验2：方法设计 
    print("设计的方法的结果（带 Attention Masking）...")
    attention_masks = process_instance_mask(mask_path)
    apply_attention_masking(pipe, attention_masks, token_indices)
    
    image_your_method = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[openpose_img, depth_img],
        controlnet_conditioning_scale=[0.7, 0.5],
        num_inference_steps=30,
        generator=torch.manual_seed(42),
        width=512,
        height=512
    ).images[0]
    image_your_method.save(f"{output_dir}/{sample_name}_your_method.png")

    print(f"✅ 生成完成！结果保存到 {output_dir}")