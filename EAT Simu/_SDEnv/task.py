import random
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
                padding = np.zeros((self.visible_len - len(self.task_queue), Task.vector_len))
                if len(self.task_queue) == 0:
                    return padding
                task_vector = np.array([task.vector for task in self.task_queue])
                return np.vstack([task_vector, padding])
                
                