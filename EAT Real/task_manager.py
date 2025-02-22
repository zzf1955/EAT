import multiprocessing
import socket
import json
from tools import send_json_to_target
from _SDEnv_real.task import Task

class DistriConfig:
    def __init__(self, node_ips: list[str], torch_port: str, master_ip: str, master_res_port: str):
        self.node_ips = node_ips
        self.torch_port = torch_port
        self.master_ip = master_ip
        self.master_res_port = master_res_port
        self.main_node_ip = self.node_ips[0] if self.node_ips else None

    def to_json(self):
        return json.dumps({
            "node_ips": self.node_ips,
            "torch_port": self.torch_port,
            "master_ip": self.master_ip,
            "master_res_port": self.master_res_port,
            "main_node_ip": self.main_node_ip
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(
            node_ips=data["node_ips"],
            torch_port=data["torch_port"],
            master_ip=data["master_ip"],
            master_res_port=data["master_res_port"]
        )

class StableDiffusionCommand:
    def __init__(self, task: Task, districonfig: DistriConfig):
        self.task = task
        self.districonfig = districonfig

    def to_json(self):
        return json.dumps({
            "task": json.loads(self.task.to_json()),
            "districonfig": json.loads(self.districonfig.to_json())
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        
        task = Task.from_json(json.dumps(data["task"]))
        districonfig = DistriConfig.from_json(json.dumps(data["districonfig"]))
        
        return cls(task=task, districonfig=districonfig)

    

class TaskManager:
    def __init__(self, slave_ips, master_ip, slave_cmd_port = 2333, master_res_port = 2333, torch_port = 2334):
        self.node_ips = slave_ips
        self.master_ip = master_ip 
        self.node_cmd_port = slave_cmd_port
        self.torch_port = torch_port
        self.master_res_port = master_res_port
        self.result_queue = multiprocessing.Manager().Queue() 

    def launch_task(self, task:Task):

        node_ids = task.node_ids
        node_ips = [self.node_ips[node_id] for node_id in node_ids]

        districonfig = DistriConfig(node_ips=node_ips,
                                    master_ip=self.master_ip,
                                    torch_port=self.torch_port,
                                    master_res_port=self.master_res_port)

        command = StableDiffusionCommand(districonfig = districonfig, task=task)

        for i,ip in enumerate(node_ips):
            send_json_to_target(data=command,target_ip=ip,target_port=self.node_cmd_port)
            listener_process = multiprocessing.Process(target=self.listen_and_receive_result, args=(command.to_json(),))
            listener_process.start()

    def listen_and_receive_result(self, cmd:StableDiffusionCommand):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.master_ip, self.master_res_port))
            sock.listen()

            print(f"Listening on {self.master_ip}:{self.master_res_port} for results...")

            while True:
                conn, addr = sock.accept()
                with conn:
                    print(f"Connected by {addr}")
                    data = b"" 

                    # Receive image data
                    while True:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet

                    if data:
                        prompt = cmd.task.info['prompt'].replace(" ", "_")
                        negative_prompt = cmd.task.info['job_info'].get('ng_prompt', '').replace(" ", "_")
                        steps = str(cmd.task.steps)
                        size = str(cmd.task.size)

                        file_name = f"{size}_{prompt}_{negative_prompt}_{steps}.png"

                        with open(file_name, 'wb') as f:
                            f.write(data)
                        
                        self.result_queue.put(file_name)
                        print(f"Received and saved result as: {file_name}")

                        break 

    def get_result(self):
        results = []
        while not self.result_queue.empty():
            file_name = self.result_queue.get()
            results.append(file_name)
        return results

def test_task_manager():
    slave_ips = ['192.168.1.101', '192.168.1.102', '192.168.1.103'] 
    master_ip = '192.168.1.100'
    master_port = 5000
    result_port = 5001
    torch_port = 5002

    task_manager = TaskManager(slave_ips, master_ip, master_port, result_port, torch_port)

    task = Task(duration = 2,
                 arrival_time = 0,
                 task_id = 1,
                 node_ids = [0],
                 co_num=1,
                 reload=True,
                 execute = True,
                 steps=30,
                 size=512*512,
                 valid = True,
                 )
    task.info['prompt'] = "orange"
    task.info['ng_prompt'] = "man!"

    task_manager.launch_task(task)

    print("Waiting for results...")

    results = task_manager.get_result()
    print("Results:", results)

if __name__ == "__main__":
    test_task_manager()
