import os
import torch

from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig

# 设置分布式环境变量
os.environ['MASTER_ADDR'] = '172.17.0.2'
os.environ['MASTER_PORT'] = '29500'
os.environ['WORLD_SIZE'] = '2'  # 2个节点
os.environ['RANK'] = '0'  # 主节点的rank为0

distri_config = DistriConfig(height=512, width=512, warmup_steps=4, mode="stale_gn")
pipeline = DistriSDPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
    local_files_only = True,
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if distri_config.rank == 0:
    image.save("astronaut.png")
