import numpy as np
from _SDEnv_real.task import Task,TaskDistriConfig, StableDiffusionCommand
from _SDEnv_real.node_manager import NodeManager
from _SDEnv_real.tools import send_command
import multiprocessing

class Node():
    vector_len = 3
    def __init__(self,
                 remain_time:float = 0,
                 cp:float = 0,
                 load_time:float = 0,
                 avaliable:bool = True,
                 node_id = 0):
        self.group = 0
        self.node_id = node_id
        self.start_time = 0
        self.remain_time = remain_time
        self.cp = cp
        self.load_time = load_time
        self.available = avaliable

    def launch_task(self, task:Task):
        self.start_time = task.start_time
        self.remain_time = task.duration
        if task.reload:
            self.remain_time+=task.load_time
        self.group = task.task_id
        self.available = False

    def working(self,time:float):
        # remain_time = 0, available = True is ctrled by cluster
        # update time
        if not self.available:
            self.remain_time -= time
        else:
            self.remain_time = 0

        # delay remain time
        if self.remain_time <= 0 and not self.available:
            print(f"node {self.node_id} delay 0.5 sec")
            self.remain_time=0.5

    def reset(self):
        self.available = True
        self.start_time = 0
        self.remain_time = 0
        self.group = 0

    @property
    def vector(self):
        return np.hstack([self.remain_time,self.group,self.available])

class Cluster():
    def __init__(self, 
                node_num:int,
                cps:list[float],
                load_times:list[float],
                state_dim = 1,
                node_ips = None,
                node_ports = [16121,16122,16123,16124]):
        nodes = []
        self.res_queue = multiprocessing.Queue()
        for cp,load_time in zip(cps,load_times):
            nodes.append(Node(cp = cp,load_time=load_time))
            nodes[-1].node_is = len(nodes)
        self.nodes = nodes
        self.nodes:list[Node]
        self.node_num = node_num
        self.state_dim = state_dim
        if(not node_ips):
            node_ips = ["127.0.0.1"]*node_num
        self.node_manager = NodeManager(node_ips = node_ips,node_ports = node_ports)

    def launch_task(self, task:Task):
        print("launch task!")
        # start task in env
        
        if not self.is_in_same_group(task.node_ids):
            task.reload = True
            self.clear_model(task=task)

        task.duration = self.predict_duration(task=task)

        for node_id in task.node_ids:
            self.nodes[node_id].launch_task(task)

        # launch real task
        p = multiprocessing.Process(target=self.node_manager.lauch_task_and_listening_res, args=(self.res_queue,task))
        p.start()

        self.update_group_id()

    def is_vaild(self, task:Task):
        avaliable_node_ids = self.get_avaliable_node_ids()
        if task.co_num<=len(avaliable_node_ids):
            return True
        return False

    def update_group_id(self):
        all_group = self.get_all_group()
        for id, group in enumerate(all_group):
            if self.nodes[group[0]].group == 0:
                continue
            for node_id in group:
                self.nodes[node_id].group = id+1

    def working(self,time:float):
        # 获取结果
        res = []
        while not self.res_queue.empty():
            msg = self.res_queue.get()
            for i in msg[0:-1]:
                self.nodes[i].remain_time = 0
                self.nodes[i].available = True
            res.append(msg[-1])

        for node in self.nodes:
            node.working(time)
        if res:
            print("receive:")
            for i in res:
                print(i.__dict__)
        return res

    def clear_model(self, task:Task):
        node_ids = task.node_ids
        unload_nodes = []
        for node_id in node_ids:
            if self.nodes[node_id].group == 0: continue
            
            id_in_same_group = self.get_same_group_node_id(self.nodes[node_id].group)
            for node_id in id_in_same_group:
                unload_nodes.append(node_id)
                self.nodes[node_id].group = 0
        self.node_manager.clear_model(unload_nodes)

    def predict_duration(self,task:Task)->float:
        # slowest_cp = self.get_slowest_cp(task.node_ids)
        duration = 0
        if task.co_num == 1:
            duration = 0.315*task.steps + (0 if not task.reload else 24)
        if task.co_num == 2:
            duration = 0.15625*task.steps + (0 if not task.reload else 24)
        if task.co_num == 4:
            duration = 0.14*task.steps + (0 if not task.reload else 32)
        return duration

    def get_same_group_node_id(self,group_id):
        return [node_id for node_id in range(self.node_num) if self.nodes[node_id].group == group_id]

    def get_slowest_cp(self,node_ids:list[int]):
        return min([self.nodes[node_id].cp for node_id in node_ids])

    def get_remain_time(self):
        return [node.remain_time for node in self.nodes]

    def get_avaliable_node_ids(self)->list[int]:
        return [i for i in range(0,self.node_num) if self.nodes[i].available]

    def get_slowest_load_time(self,node_ids:list[int]):
        return max([self.nodes[node_id].load_time for node_id in node_ids])

    def get_avaliable_group(self)->list[list[int]]:
        avaliable_node_id = self.get_avaliable_node_ids()
        group_dict = {}
        for node_id in avaliable_node_id:
            group = self.nodes[node_id].group
            if group == 0:
                continue
            if group not in group_dict:
                group_dict[group] = []
            group_dict[group].append(node_id)
        
        return list(group_dict.values())

    def get_all_group(self):
        group_dict = {}
        for node_id in range(self.node_num):
            group = self.nodes[node_id].group
            remain_time = self.nodes[node_id].remain_time
            if (group,remain_time) not in group_dict:
                group_dict[(group,remain_time)] = []
            group_dict[(group,remain_time)].append(node_id)
        
        return list(group_dict.values())

    def is_avaliable(self,node_ids:list[int]):
        return all([self.nodes[node_id].available for node_id in node_ids])

    def is_in_same_group(self,node_ids:list[int]):
        group0 = self.nodes[node_ids[0]].group
        if group0 == 0:
            return False
        group0_index = [node_id for node_id in range(self.node_num) if self.nodes[node_id].group == group0]
        return group0_index == node_ids
 
    def all_done(self):
        return max([node.remain_time for node in self.nodes]) == 0

    def reset(self):
        for node in self.nodes:
            node.reset()

    @property
    def vector(self):
        if self.state_dim == 1:
            return np.hstack([node.vector for node in self.nodes])
        if self.state_dim == 2:
            return np.stack([node.vector for node in self.nodes])
