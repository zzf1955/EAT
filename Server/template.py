import numpy as np
import json

class Task():
    vector_len = 3
    def __init__(self,
                 state_dim = 1,
                 duration: float = 0,
                 arrival_time: float = 0,
                 task_id: int = -1,
                 node_ids: list[int] = [],
                 co_num=1,
                 reload=False,
                 execute = True,
                 steps=30,
                 size=512*512,
                 valid = True,
                 real_arrival_time = 0,
                 finish_time = 0,
                 start_time = 0,
                 load_time = 0,
                 info: dict={},**kwargs):

        # Assigning each parameter to a member variable
        self.duration = duration
        self.arrival_time = arrival_time
        self.task_id = task_id
        self.node_ids = node_ids
        self.co_num = co_num
        self.reload = reload
        self.steps = steps
        self.size = size
        self.info = info
        self.execute = execute
        self.valid = valid
        self.start_time = -1
        self.load_time = 0
        self.real_arrival_time = 0,
        self.finish_time = 0,
        self.start_time = 0,
    @property
    def vector(self):
        return np.hstack([self.arrival_time,self.size/(512*512),self.co_num])

    def to_json(self):
        # Convert the object's attributes to a dictionary and then to a JSON string
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        # Convert JSON string to a dictionary and use it to create a Task instance
        data = json.loads(json_str)
        # Create a new Task instance using the unpacked dictionary
        return cls(**data)

class TaskDistriConfig:
    def __init__(self, node_ips: list[str], torch_port: str, master_ip: str, master_res_port: str, node_id:str):
        self.node_ips = node_ips
        self.torch_port = str(torch_port)
        self.master_ip = master_ip
        self.master_res_port = str(master_res_port)
        self.main_node_ip = self.node_ips[0] if self.node_ips else None
        self.node_id = str(node_id)

    def to_json(self):
        # 将对象的所有属性序列化为 JSON 字符串
        return json.dumps({
            "node_ips": self.node_ips,
            "torch_port": self.torch_port,
            "master_ip": self.master_ip,
            "master_res_port": self.master_res_port,
            "main_node_ip": self.main_node_ip,
            "node_id":self.node_id
        })

    def __eq__(self, other):
        if not isinstance(other, TaskDistriConfig):
            return False
        return (
            self.node_ips == other.node_ips and
            self.torch_port == other.torch_port and
            self.master_ip == other.master_ip and
            self.master_res_port == other.master_res_port and
            self.node_id == other.node_id
        )

    @classmethod
    def from_json(cls, json_str):
        # 反序列化 JSON 字符串为一个字典
        data = json.loads(json_str)
        # 从字典中提取需要的字段，并创建 DistriConfig 实例
        return cls(
            node_ips=data["node_ips"],
            torch_port=data["torch_port"],
            master_ip=data["master_ip"],
            master_res_port=data["master_res_port"],
            node_id = data["node_id"]
        )

class StableDiffusionCommand:
    def __init__(self, task: Task, districonfig: TaskDistriConfig):
        self.task = task
        self.districonfig = districonfig

    def to_json(self):
        # 使用 Task 和 DistriConfig 的 to_json 方法分别序列化每个对象
        return json.dumps({
            "task": json.loads(self.task.to_json()),         # 确保嵌套字典格式
            "districonfig": json.loads(self.districonfig.to_json())
        })

    @classmethod
    def from_json(cls, json_str):
        # 解析 JSON 字符串为字典
        data = json.loads(json_str)
        
        # 分别反序列化 Task 和 DistriConfig 部分
        task = Task.from_json(json.dumps(data["task"]))
        districonfig = TaskDistriConfig.from_json(json.dumps(data["districonfig"]))
        
        # 创建并返回 StableDiffusionCommand 实例
        return cls(task=task, districonfig=districonfig)
