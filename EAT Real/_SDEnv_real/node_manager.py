import numpy as np
from _SDEnv_real.task import Task, TaskDistriConfig, StableDiffusionCommand
from _SDEnv_real.tools import send_command, receive_result
import socket
import multiprocessing
import time

class NodeManager:
    def __init__(self,
                 node_ips:list[str],
                 node_ports:list[int]):
        self.node_ips = node_ips
        self.node_ports = node_ports

    def lauch_task_and_listening_res(self, res_queue:multiprocessing.Queue, task:Task, file_name = None):
        start_time = time.time()
        if file_name == None:
            file_name = f"{task.co_num}_{task.steps}_{task.info['prompt']}_{task.info['ng_prompt']}_{task.task_id}.png"
        res_port = send_command(task = task, node_ips = self.node_ips,node_ports = self.node_ports)
        res_name = receive_result(self.node_ips[task.node_ids[0]],res_port,file_name)
        finish_time = time.time()
        task.finish_time = finish_time
        task.start_time = start_time
        # send complete info
        res_queue.put(task.node_ids+[task])
        return res_name
    
    def clear_model(self, node_ids:list[int]):
        task = Task()
        task_districonfig = TaskDistriConfig(
            node_ips=[self.node_ips[i] for i in node_ids],
            torch_port=0,
            master_ip="127.0.0.1",
            master_res_port=0,
            node_id = 0
        )
        command = StableDiffusionCommand(task=task,districonfig=task_districonfig)
        for node_id in node_ids:
            target_ip = self.node_ips[node_id]

            json_data = command.to_json()
            server_address = (target_ip, self.node_ports[node_id])
            success = False
            try_times = 0
            while not success:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    try:
                        sock.connect(server_address)
                        sock.sendall(json_data.encode('utf-8'))
                        print(f"stop sent to server {node_id} successfully. try {try_times} times")
                        success = True
                    except ConnectionRefusedError as e:
                        if try_times == 0:
                            pass
                            #print(f"Error: Connection failed to {node_id} - {str(e)}")
                            #print("Start Retry")
                        time.sleep(0.1)
