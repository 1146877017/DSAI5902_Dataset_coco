import os

# ====================== 模型与路径配置 ======================
# 基础底模（SD1.5架构二次元底模，如Anything V4.5）
BASE_MODEL_PATH = "./anything-v4.5.safetensors"

# ControlNet 预训练模型
CONTROLNET_OPENPOSE = "lllyasviel/sd-controlnet-openpose"
CONTROLNET_DEPTH = "lllyasviel/sd-controlnet-depth"

# LoRA 存放目录
LORA_DIR = "./lora_weights"

# 输出根目录
OUTPUT_ROOT = "./synthetic_test_results"
# 测试条件（pose/depth/mask）存放目录
CONDITION_DIR = "./test_conditions"

# ====================== 生成全局参数 ======================
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 768
NUM_INFERENCE_STEPS = 28
CFG_SCALE = 7.0
# 全局固定种子，所有实验组共用，消除随机性干扰
GLOBAL_SEED = 42
LORA_WEIGHT = 0.8

# ====================== 角色配置 ======================
CHARACTER_CONFIGS = {
    "Asuna_Stacia": {
        "file": "asuna_(stacia)-v1.5.safetensors",
        "trigger": "stacia",
        "desc": "white dress, armor, white thighhighs"
    },
    "Toga_Himiko": {
        "file": "TogaHimiko-01.safetensors",
        "trigger": "Himiko Toga",
        "desc": ""
    },
    "Rinne_Byakuya": {
        "file": "Byakuya.safetensors",
        "trigger": "Byakuyadef",
        "desc": "black sailor uniform, black hair, long hair"
    },
    "Mouri_Ran": {
        "file": "Mouri.safetensors",
        "trigger": "mouriranai",
        "desc": "school uniform, blue jacket, green necktie"
    }
}

# 双角色配对（所有两两组合）
CHARACTER_PAIRS = [
    ("Asuna_Stacia", "Toga_Himiko"),
    ("Asuna_Stacia", "Rinne_Byakuya"),
    ("Asuna_Stacia", "Mouri_Ran"),
    ("Toga_Himiko", "Rinne_Byakuya"),
    ("Toga_Himiko", "Mouri_Ran"),
    ("Rinne_Byakuya", "Mouri_Ran")
]

# ====================== 场景与背景配置 ======================
# 叙事场景（对应提案 story-based scenarios）
SCENES = ["side_by_side", "handshake", "front_back"]
# 背景环境（至少3种，对应提案要求）
BACKGROUNDS = ["outdoor_park", "indoor_room", "city_street"]

# ====================== 通用负向提示词 ======================
NEGATIVE_PROMPT = """
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits,
cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark,
blurry, deformed, mutated, ugly, disfigured
"""