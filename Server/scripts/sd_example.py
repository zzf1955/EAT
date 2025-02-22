import torch
from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig

distri_config = DistriConfig(height=768, width=1024, warmup_steps=4)

pipeline = DistriSDPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    local_files_only = True
)

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    generator=torch.Generator(device="cuda").manual_seed(233),
).images[0]
if distri_config.rank == 0:
    image.save("astronaut.png")
