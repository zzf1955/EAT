import tools
import subprocess
from template import StableDiffusionCommand
import re

def build_torch_command(cmd: StableDiffusionCommand):
    # Get the number of nodes (num_slaves)
    node_num = len(cmd.districonfig.node_ips)

    # Get the node rank (slave_id - 1)
    node_id = cmd.districonfig.node_id

    # Format the torchrun command
    torch_command = [
        "torchrun",
        f"--nnodes={node_num}",
        f"--node_rank={node_id}",
        "--nproc_per_node=1",
        f"--master_addr={cmd.districonfig.master_ip}",
        f"--master_port={cmd.districonfig.torch_port}",
        "run_sd.py",  # Path to the script
        f"{cmd.to_json()}"  # Pass the parameters in JSON format
    ]

    return torch_command

def run_node(command_port):
    while True:
        cmd = tools.receive_task(command_port)
        while(cmd.districonfig.torch_port == '0'):
            cmd = tools.receive_task(command_port)
        torch_command = build_torch_command(cmd=cmd)
        print(f"receive torch_command in main:{torch_command}")

        result = subprocess.run(torch_command, capture_output=False, text=True, check=True, encoding='utf-8')
        print(f"get sub process output:\n{result.stdout}")

if __name__ == '__main__':
    command_port = 16122
    run_node(16122)
