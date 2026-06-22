import torch
print("torch version:", torch.__version__)
print("has float8_e8m0fnu:", hasattr(torch, 'float8_e8m0fnu'))

import transformers
print("transformers version:", transformers.__version__)

import diffusers
print("diffusers version:", diffusers.__version__)