import torch
from diffusers import StableDiffusionPipeline

# 把整个模型安全地下载到系统的默认缓存目录中
pipe = StableDiffusionPipeline.from_pretrained("Lykon/AbsoluteReality", torch_dtype=torch.float16)
print("模型下载完成！")