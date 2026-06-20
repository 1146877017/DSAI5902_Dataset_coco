import os
import torch
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from config import *

def load_condition_images(condition_dir):
    """加载单样本的控制条件图"""
    pose = Image.open(os.path.join(condition_dir, "pose.png")).convert("RGB")
    depth = Image.open(os.path.join(condition_dir, "depth.png")).convert("RGB")
    return pose, depth

def build_prompt(char_a_name, char_b_name, background):
    """构建统一格式的提示词"""
    char_a = CHARACTER_CONFIGS[char_a_name]
    char_b = CHARACTER_CONFIGS[char_b_name]
    return (
        f"left character: {char_a['trigger']}, {char_a['desc']}, "
        f"right character: {char_b['trigger']}, {char_b['desc']}, "
        f"{background} background, full body, masterpiece, best quality, anime style"
    )

def init_baseline_pipeline(baseline_type):
    """根据基线类型初始化对应管线"""
    if baseline_type == "baseline1":
        # 基线1：纯文本，无ControlNet
        pipe = StableDiffusionPipeline.from_single_file(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    elif baseline_type == "baseline2":
        # 基线2：单ControlNet（仅OpenPose）
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_OPENPOSE, torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            BASE_MODEL_PATH,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    elif baseline_type == "baseline3":
        # 基线3：多ControlNet（OpenPose + Depth）
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
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

def load_dual_lora(pipe, char_a_name, char_b_name):
    """加载双角色LoRA，使用适配器机制避免权重污染"""
    char_a = CHARACTER_CONFIGS[char_a_name]
    char_b = CHARACTER_CONFIGS[char_b_name]

    pipe.load_lora_weights(
        os.path.join(LORA_DIR, char_a["file"]),
        adapter_name="char_a"
    )
    pipe.load_lora_weights(
        os.path.join(LORA_DIR, char_b["file"]),
        adapter_name="char_b"
    )
    pipe.set_adapters(["char_a", "char_b"], adapter_weights=[LORA_WEIGHT, LORA_WEIGHT])
    return pipe

def run_single_baseline(baseline_type):
    """运行单组基线的所有样本"""
    print(f"===== Running {baseline_type} =====")
    pipe = init_baseline_pipeline(baseline_type)
    generator = torch.Generator("cuda").manual_seed(GLOBAL_SEED)

    for scene in SCENES:
        for bg in BACKGROUNDS:
            for (char_a, char_b) in CHARACTER_PAIRS:
                condition_dir = os.path.join(CONDITION_DIR, scene, bg, f"{char_a}_{char_b}")
                save_dir = os.path.join(OUTPUT_ROOT, baseline_type, scene, bg, f"{char_a}_{char_b}")
                os.makedirs(save_dir, exist_ok=True)

                # 加载LoRA
                pipe = load_dual_lora(pipe, char_a, char_b)
                prompt = build_prompt(char_a, char_b, bg)

                if baseline_type == "baseline1":
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=CFG_SCALE,
                        height=IMAGE_HEIGHT,
                        width=IMAGE_WIDTH,
                        generator=generator
                    ).images[0]
                else:
                    pose_img, depth_img = load_condition_images(condition_dir)
                    control_imgs = [pose_img] if baseline_type == "baseline2" else [pose_img, depth_img]
                    image = pipe(
                        prompt=prompt,
                        image=control_imgs,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=CFG_SCALE,
                        height=IMAGE_HEIGHT,
                        width=IMAGE_WIDTH,
                        generator=generator
                    ).images[0]

                image.save(os.path.join(save_dir, "result.png"))
                print(f"Saved: {save_dir}/result.png")

                # 卸载LoRA，避免下一样本残留
                pipe.unload_lora_weights()

    del pipe
    torch.cuda.empty_cache()
    print(f"{baseline_type} completed.\n")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # 依次运行三组基线
    run_single_baseline("baseline1")
    run_single_baseline("baseline2")
    run_single_baseline("baseline3")
    print("All baselines finished.")

if __name__ == "__main__":
    main()