import os
import torch

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig

# 设置分布式环境变量
os.environ['MASTER_ADDR'] = '172.17.0.2'  # 主节点的 IP 地址
os.environ['MASTER_PORT'] = '29500'  # 通信端口
os.environ['WORLD_SIZE'] = '2'  # 总的节点数
os.environ['RANK'] = '0'  # 当前节点的 rank，主节点为 0

# 配置分布式
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)

# 加载模型
pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    local_files_only=True,
)

# 设置进度条的配置
pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

# 生成图像
image = pipeline(
    prompt="Oranges in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]

# 保存图像到文件
if distri_config.rank == 0:
    image.save("oranges_in_jungle.png")
