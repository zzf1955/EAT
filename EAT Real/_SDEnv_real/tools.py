import time
import socket
import json
import csv
from _SDEnv_real.task import Task, TaskDistriConfig, StableDiffusionCommand

def receive_result(ip, port, output_file):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((ip, int(port)))
            sock.listen(1)
            print(f"Listening for incoming data on {ip}:{port}")

            conn, addr = sock.accept()
            with conn:
                print(f"Connected by {addr}")

                with open(output_file, 'wb') as f:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        f.write(data)

                print(f"Data received and saved to {output_file}")
    except Exception as e:
        print(f"Error receiving data on {ip}:{port}: {e}")

def get_new_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    print(f"get new port in main node:{port}")
    return port

def send_command(task:Task,node_ips:list[str], node_ports = list[int], task_type = "task"):
    master_res_port = get_new_port()
    if task_type == "task":
        task_districonfig = TaskDistriConfig(
            node_ips=[node_ips[i] for i in task.node_ids],
            torch_port=get_new_port(),
            master_ip=node_ips[0],
            master_res_port=master_res_port,
            node_id = 0
        )
    if task_type == "stop":
        task_districonfig = TaskDistriConfig(
            node_ips=[node_ips[i] for i in task.node_ids],
            torch_port=0,
            master_ip=node_ips[0],
            master_res_port=0,
            node_id = 0
        )
    _send_command(task_type=task_type,
                task=task,
                distri_config=task_districonfig,
                node_ports=node_ports)
    return master_res_port

def _send_command(task_type, task, distri_config, node_ports):
    for i, target_ip in enumerate(distri_config.node_ips):
        distri_config.node_id = i
        command = StableDiffusionCommand(task=task, districonfig=distri_config)
        json_data = command.to_json()
        server_address = (target_ip, node_ports[task.node_ids[i]])
        success = False
        try_times = 0
        while not success:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.connect(server_address)
                    sock.sendall(json_data.encode('utf-8'))
                    print(f"{task_type} sent to server {task.node_ids[i]} successfully. try {try_times} times")
                    success = True
                except ConnectionRefusedError as e:
                    print(f"Error: Connection failed to {task.node_ids[i]} - {str(e)} - {task_type}")
                    print("Start Retry")
                    pass
                    time.sleep(0.1)
