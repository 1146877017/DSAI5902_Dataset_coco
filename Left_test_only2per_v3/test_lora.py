import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ====================== 配置项（根据你的本地路径修改） ======================
# 基础模型路径（SD1.5系列，建议用二次元底模，如 Anything-V4.5）
BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # 可替换为本地模型路径
# LoRA 文件存放目录
LORA_DIR = "./lora_weights"
# 结果保存目录
OUTPUT_DIR = "./lora_validation_results"
# 生成参数
IMAGE_SIZE = (512, 768)  # 竖版人物，适配SD1.5
NUM_INFERENCE_STEPS = 28
CFG_SCALE = 7.0
SEED_LIST = [42, 123, 456]  # 固定3个种子，验证生成稳定性
LORA_WEIGHT = 0.85  # LoRA权重，可根据效果微调


# 4个LoRA的完整配置（文件名 + 触发词 + 正向描述）
LORA_CONFIGS = [
    {
        "name": "Asuna_Stacia",
        "file": "asuna_(stacia)-v1.5.safetensors",
        "trigger": "stacia",
        "prompt_supplement": "1girl, solo, full body, white dress, armor, white thighhighs, masterpiece, best quality, anime style, detailed face"
    },
    {
        "name": "Toga_Himiko",
        "file": "TogaHimiko-01.safetensors",
        "trigger": "Himiko Toga",
        "prompt_supplement": "1girl, solo, full body, masterpiece, best quality, anime style, detailed face"
    },
    {
        "name": "Rinne_Byakuya",
        "file": "Byakuya.safetensors",
        "trigger": "Byakuyadef",
        "prompt_supplement": "1girl, solo, full body, black sailor uniform, black hair, long hair, masterpiece, best quality, anime style, detailed face"
    },
    {
        "name": "Mouri_Ran",
        "file": "Mouri.safetensors",
        "trigger": "mouriranai, mouri ran",
        "prompt_supplement": "1girl, solo, full body, brown hair, blue eyes, long hair, school uniform, blue jacket, green necktie, pleated skirt, masterpiece, best quality, anime style, detailed face"
    }
]

# 通用负向提示词
NEGATIVE_PROMPT = """
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, mutated, ugly, disfigured, multiple girls
"""

# ==========================================================================

def init_pipeline():
    """加载基础生成管线"""
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    # 开启xformers加速（可选，需安装xformers）
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    return pipe

def test_single_lora(pipe, lora_cfg, save_dir):
    """测试单个LoRA，生成多张图并保存"""
    lora_path = os.path.join(LORA_DIR, lora_cfg["file"])
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载当前LoRA
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=LORA_WEIGHT)
    
    # 拼接正向提示词
    prompt = f"{lora_cfg['trigger']}, {lora_cfg['prompt_supplement']}"
    
    print(f"Testing: {lora_cfg['name']}")
    for seed in SEED_LIST:
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=CFG_SCALE,
            height=IMAGE_SIZE[1],
            width=IMAGE_SIZE[0],
            generator=generator
        ).images[0]
        
        # 保存图片
        save_path = os.path.join(save_dir, f"{lora_cfg['name']}_seed{seed}.png")
        image.save(save_path)
        print(f"  Saved: {save_path}")
    
    # 卸载当前LoRA，避免影响下一个测试
    pipe.unfuse_lora()
    pipe.unload_lora_weights()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe = init_pipeline()
    
    for lora_cfg in LORA_CONFIGS:
        save_subdir = os.path.join(OUTPUT_DIR, lora_cfg["name"])
        test_single_lora(pipe, lora_cfg, save_subdir)
    
    print("All LoRA tests completed. Results saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()