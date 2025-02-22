import random
import numpy as np
import json
import time

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
                 quality = 0,
                 quality_reward = 0,
                 time_reward = 0,
                 info: dict={}):
        
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
        self.real_arrival_time = real_arrival_time,
        self.finish_time = finish_time,
        self.start_time = start_time,
        self.quality = quality
        self.quality_reward = quality_reward
        self.time_reward = time_reward


    @property
    def vector(self):
        return np.hstack([self.arrival_time,self.size/(512*512),self.co_num])

    def to_json(self):
        self.duration = float(self.duration)
        self.steps = int(self.steps)
        self.arrival_time = float(self.arrival_time)
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        # Convert JSON string to a dictionary and use it to create a Task instance
        data = json.loads(json_str)
        # Create a new Task instance using the unpacked dictionary
        return cls(**data)

class TaskGenerator():
    def __init__(self,
                 fixed_steps,
                 fixed_co_num,
                 fixed_size = True,
                 max_steps = 64,
                 co_num = [1,2,4,8],
                 size_option = [512,768,1024]):
        self.fixed_steps = fixed_steps
        self.fixed_co_num = fixed_co_num
        self.fixed_size = fixed_size
        self.size_option = size_option
        self.max_steps = max_steps
        self.co_num = co_num
        self.task_cnt = 0

    def reset(self):
        self.task_cnt = 0

    def get_new_task(self):
        task = Task()
        task.real_arrival_time = time.time()
        self.task_cnt+=1
        task.task_id = self.task_cnt
        if self.fixed_steps:
            task.steps = random.random()*self.max_steps
        if self.fixed_co_num:
            task.co_num = random.choice(self.co_num)
        if self.fixed_size:
            task.size = 512*512
        else:
            task.size = random.choice(self.size_option)*random.choice(self.size_option)

        task.info['prompt'] = "orange"
        task.info['ng_prompt'] = "man!"

        return task

class TaskQueue():

    def __init__(self,
                 task_generator:TaskGenerator,
                 visible_len:int = 10,
                 init_job_num:int = 10,
                 state_dim = 1):
        self.visible_len = visible_len
        self.task_queue = []
        self.task_queue:list[Task]
        for i in range(init_job_num):
            self.task_queue.append(task_generator.get_new_task())
        self.state_dim = state_dim

    def empty(self):
        return len(self.task_queue) == 0

    def is_vaild_id(self,id):
        return id<=len(self.task_queue)

    def get_avg_waiting_time(self, current_time:float)->float:
        task_num = min(self.visible_len,len(self.task_queue))
        if task_num == 0:
            return 0
        waiting_time = sum([(current_time-task.arrival_time) for task in self.task_queue[:task_num]])/task_num
        return waiting_time

    def remove_task_id(self,id = -1):
        if not self.is_vaild_id(id):
            assert("task index error! not in queue")
        self.task_queue.pop(id)

    def remove_task(self,task:Task):
        if not(task in self.task_queue):
            assert("task not in queue")
        self.task_queue.remove(task)

    @property
    def vector(self):
        if self.state_dim == 1:
            if self.visible_len<=len(self.task_queue):
                return np.hstack(np.ravel([task.vector for task in self.task_queue[:self.visible_len]]))
            else:
                return np.hstack([np.ravel([task.vector for task in self.task_queue]),
                                [0]*Task.vector_len*(self.visible_len-len(self.task_queue))])
        if self.state_dim == 2:
            if self.visible_len<=len(self.task_queue):
                return np.stack([task.vector for task in self.task_queue[:self.visible_len]])
            else:
                # 生成填充的零矩阵，每行都是 [0, 0, 0]
                padding = np.zeros((self.visible_len - len(self.task_queue), Task.vector_len))
                # 将已有的 task.vector 与填充矩阵在行方向上进行拼接
                if len(self.task_queue) == 0:
                    return padding
                task_vector = np.array([task.vector for task in self.task_queue])
                return np.vstack([task_vector, padding])
                
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
