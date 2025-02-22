import torch
from distrifuser.pipelines import DistriSDPipeline
from distrifuser.utils import DistriConfig
import argparse
from template import StableDiffusionCommand
from tools import receive_task, send_result
import torch.distributed as dist
import sys
import gc

def get_cmd()->StableDiffusionCommand:
    return StableDiffusionCommand.from_json(sys.argv[1])

def run_sd():

    cmd = get_cmd()
    distri_config = DistriConfig(height=1440, width=1440, warmup_steps=4, mode="stale_gn")
    pipeline = DistriSDPipeline.from_pretrained(
        distri_config = distri_config,
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4",
        local_files_only = True,
    )

    while(True):

        image = pipeline(
            prompt=cmd.task.info['prompt'],
            negative_prompt=cmd.task.info['ng_prompt'],
            generator=torch.Generator(device="cuda").manual_seed(233),
            num_inference_steps = cmd.task.steps
        ).images[0]
        file_name = f"{cmd.task.info['prompt'].replace(' ', '_')}__{cmd.task.info['ng_prompt'].replace(' ', '_')}__{cmd.task.steps}.png"
        #file_name = "output.png"
        image.save(file_name)
        if(cmd.districonfig.node_id == "0"):
                send_result(cmd.districonfig.master_ip,cmd.districonfig.master_res_port,file_name=file_name)
        if(len(cmd.districonfig.node_ips)>1):
            dist.barrier()
        print("task finish!")
        new_cmd = receive_task(16122)
        print(f"old districonfig:{cmd.task.node_ids}")
        print(f"new dustriconfig:{new_cmd.task.node_ids}")
        if cmd.task.node_ids != new_cmd.task.node_ids :
            if len(cmd.districonfig.node_ips) > 1:
                print("waiting for other node")
                dist.barrier()
            print("start destroy group")
            torch.cuda.synchronize()
            del pipeline
            gc.collect()
            dist.destroy_process_group()
            print("sub process finish!")
            return
        cmd = new_cmd

run_sd()
