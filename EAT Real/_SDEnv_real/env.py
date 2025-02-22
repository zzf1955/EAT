from tianshou.env import DummyVectorEnv
from tabulate import tabulate
import numpy as np
import time
import os
import csv
import math
import random
import gymnasium as gym
from gymnasium import spaces
from _SDEnv_real.cluster import Cluster, Node
from _SDEnv_real.task import Task, TaskQueue, TaskGenerator

class AIGCEnv(gym.Env):
    def __init__(self, 
                queue_len, 
                node_num,
                cp:list[float],
                load_time=10,
                max_draw_steps=32,
                co_num=[1, 2, 4], 
                w_t=1,
                w_q=5,
                w_tt = 0.2,
                task_num=32,
                time_limite=360,
                steps_limite=100000,
                fixed_co=True,
                fixed_steps=False,
                fixed_size = True,
                seed = 233,
                state_dim = 1,
                node_ips = None,
                node_ports = None,
                min_task_arrival_rate = 0.0,
                max_task_arrival_rate = 1.0,
                T = 2,
                t = 0.5
                ):
        super(AIGCEnv, self).__init__()
        self.t = t;
        self.set_seed(seed=seed)
        self.state_dim = state_dim

        # reward
        self.wt = w_t
        self.wq = w_q
        self.wtt = w_tt
        
        # limite
        self.task_num = task_num
        self.time_limite = time_limite
        self.steps_limite = steps_limite

        self.min_task_arrival_rate = min_task_arrival_rate
        self.max_task_arrival_rate = max_task_arrival_rate
        self.task_arrival_rate = min_task_arrival_rate  # Initial task arrival rate
        self.T = T

        # address
        if(not node_ips):
            node_ips = ["127.0.0.1"]*node_num
        if(not node_ports):
            node_ports = [int("1612" + str(i)) for i in range(1,node_num+1)]

        self.node_ips = node_ips
        self.node_ports = node_ports

        self.task_generator = TaskGenerator(
            fixed_co_num = fixed_co, 
            fixed_steps = fixed_steps,
            fixed_size = fixed_size,
            co_num=co_num)
        
        self.task_queue = TaskQueue(
            self.task_generator, 
            visible_len=queue_len, 
            init_job_num=queue_len,
            state_dim=state_dim)
        
        self.cluster = Cluster(
            node_num=node_num,
            cps = cp,
            load_times=[load_time]*node_num,
            state_dim = self.state_dim,
            node_ips=node_ips,
            node_ports=node_ports)
        
        self.max_draw_steps = max_draw_steps
        self.current_time = 1
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + queue_len,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(queue_len*Task.vector_len + node_num*Node.vector_len,), dtype=np.float32)
        self.statistics = []
        self.start_time = -1
        self.end_time = 0
        self.arrived_task_cnt = 0


    def reset(self, SEED = 233):

        self.set_seed(seed=SEED)
        self.task_generator.reset()

        print("waiting for task finished")
        time.sleep(1)
        task = Task()
        node_ids = [i for i in range(len(self.node_ips))]
        task.node_ids = node_ids
        self.cluster.clear_model(task)
        self.cluster.reset()
        
        self.task_queue = TaskQueue(
            self.task_generator, 
            visible_len=self.task_queue.visible_len, 
            init_job_num=self.task_queue.visible_len,
            state_dim=self.state_dim)
        
        self.current_time = 1
        self.current_step = 0
        # self.tt_quality = 0
        # self.tt_time = 0

        return self.get_obs()

    def add_pre_statistic(self, task: Task):
        steps = task.steps
        if steps < 5:
            quality = 0
        else:
            a = 0.16
            b = -0.02
            base = 6
            quality = a * np.log(steps) / np.log(base) + b
        if steps > 30:
            quality = 0.25
        quality_reward = self.wq * quality

        avg_waiting_time = self.task_queue.get_avg_waiting_time(self.current_time)
        task.quality = quality
        task.quality_reward = quality_reward
        time_reward = self.wt / (
            self.wtt * avg_waiting_time +
            (task.duration + self.current_time - task.arrival_time)
        )
        time_reward =min(10,time_reward)
        task.time_reward = time_reward

    def add_lat_statistic(self, task: Task):
        print(f"old data:{task.__dict__}")
        task.waiting_time = task.start_time - task.real_arrival_time
        task.running_time = task.finish_time-task.start_time
        task.response_time = task.waiting_time + task.running_time
        print(f"new data:{task.__dict__}")

    def step(self, action):
        t = self.t
        if self.start_time == -1:
            self.start_time = time.time()

        self.current_step += 1
        truncated = False
        done = False
        if self.current_time<=self.time_limite:
            task = self.get_task(action)

            if (not task.valid) or (not task.execute):
                self.update_time(t)
                tmp_time = time.time()
                res = self.cluster.working(t)
                print(f"cluster working for {time.time()-tmp_time}s")
                for res_task in res:
                    self.add_lat_statistic(res_task)
                    self.statistics.append(res_task)
            else:
                self.add_pre_statistic(task=task)
                self.cluster.launch_task(task=task)
                task.start_time = time.time()
                self.task_queue.remove_task(task)   
        else:
            self.update_time(t)
            res = self.cluster.working(t)
            for res_task in res:
                self.add_lat_statistic(res_task)
                self.statistics.append(res_task)
        #if(self.current_step>self.steps_limite) or\
        #    self.current_time>self.time_limite:
        #    truncated = True
        #if(self.task_generator.task_cnt >= self.task_num and self.task_queue.empty() and self.cluster.all_done()):
        #    done = True

        if self.current_time>self.time_limite or (self.current_step>self.steps_limite) :
            truncated = True
        if(self.task_generator.task_cnt>=self.task_num and self.task_queue.empty() and self.cluster.all_done()):
            done = True

        reward = self.calculate_reward(task=task)
        self.render()
        obs = self.get_obs()
        return obs, reward, done, truncated, {}

    def get_obs(self):
        if self.state_dim == 1:
            return np.round(np.hstack([self.task_queue.vector, self.cluster.vector]))
        if self.state_dim == 2:
            return np.round(np.vstack([self.task_queue.vector, self.cluster.vector]))


    def get_task(self,action)->Task:
        """
        action format:
        [execute, step, co1..n, task1..n]
        """
        action = (action+1)/2

        step_action = execute_action = co_action = task_action = None
        co_list = self.task_generator.co_num
        fixed_steps = self.task_generator.fixed_steps
        fixed_co_num = self.task_generator.fixed_co_num

        pos = 0
        execute_action = action[pos]
        pos+=1
        if not fixed_steps:
            step_action = action[pos]
            pos+=1
        if not fixed_co_num:
            co_action = action[pos:pos+len(co_list)]
            pos+=len(co_list)
        task_action = action[pos:]
        pos+=self.task_queue.visible_len

        if not(pos == len(action)):
            assert("build action error")
        
        if np.argmax(task_action)>=len(self.task_queue.task_queue):
            task = Task(valid=False)
            return task
    
        task = self.task_queue.task_queue[np.argmax(task_action)]

        if step_action:
            task.steps = max(int(step_action*self.max_draw_steps),1)

        if co_action:
            task.co_num = np.argmax(co_action)

        if execute_action:
            if execute_action < 0.5:
                task.execute = False
            else:
                task.execute = True

        task.node_ids = self.cluster.get_avaliable_node_ids()[:task.co_num]
        for group in self.cluster.get_avaliable_group():
            if len(group) == task.co_num:
                task.node_ids = group
                break

        task.valid = self.cluster.is_vaild(task = task)
        if task.valid:
            task.duration = self.cluster.predict_duration(task=task)
            task.load_time = self.cluster.get_slowest_load_time(task.node_ids)
        return task

    def calculate_reward(self, task:Task):

        # None Action or Invalid Action则返回平均等待时间
        if (not task.execute) or (not task.valid):
            return 0

        if not task.duration:
            assert("missing duration")
        if not task.co_num:
            assert("missing co num")
        avg_waiting_time = self.task_queue.get_avg_waiting_time(self.current_time)
        steps = task.steps
        quality = 0
        if steps < 5:
            quality = 0
        else:
            a = 0.16
            b = -0.02
            base = 6
            quality = a * np.log(steps) / np.log(base) + b
        if steps > 30:
            quality = 0.25

        time_reward = self.wt / (
            self.wtt * avg_waiting_time +
            (task.duration + self.current_time - task.arrival_time)
        )
        time_reward =min(10,time_reward)

        quality_reward = self.wq * quality
        return time_reward + quality_reward

    def update_task_arrival_rate(self):
            # 改为基于时间的周期性变化（原基于 task_cnt）
            period_position = (self.current_time % self.time_limite) / self.time_limite
            cos_value = math.cos(2 * math.pi * period_position * self.T)
            self.task_arrival_rate = self.min_task_arrival_rate + \
                (self.max_task_arrival_rate - self.min_task_arrival_rate) * (0.5 * (cos_value + 1))

    def update_time(self, working_time):

        #self.update_task_arrival_rate()

        self.end_time = time.time()
        current_time = time.time()
        if not self.task_generator.task_cnt>=self.task_num and self.current_time<self.time_limite:
            for i in range(int(working_time*10)):
                if random.random()>self.task_arrival_rate/10:
                    continue
                task = self.task_generator.get_new_task()
                task.arrival_time = self.current_time + i/10
                self.task_queue.task_queue.append(task)

        print(f"working for {self.end_time - self.start_time} sec")
        if(self.end_time - self.start_time>working_time):
            print("--------\ntime_out!\n--------")
        else:
            print(f"waiting for {working_time-(self.end_time - self.start_time)} sec")
            time.sleep(working_time-(self.end_time - self.start_time))
        self.current_time+=working_time
        self.start_time = time.time()

    def set_seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def render(self):
        # Cluster Nodes Table
        node_headers = ["Node ID", "Remain Time", "Group", "Available"]
        node_table = [
            [node.node_id, node.remain_time, node.group, node.available]
            for node in self.cluster.nodes
        ]

        # 转置 Cluster Nodes 表格
        transposed_node_table = list(zip(*([node_headers] + node_table)))
        print("Cluster Nodes Information:")
        print(tabulate(transposed_node_table, headers="firstrow", tablefmt="grid"))

        # Task Queue Table
        task_headers = ["Task ID", "Co-Number", "Arrival Time"]
        task_table = [
            [task.task_id, task.co_num, task.arrival_time]
            for task in self.task_queue.task_queue[0:min(len(self.task_queue.task_queue),self.task_queue.visible_len)] 
        ]

        # 转置 Task Queue 表格
        transposed_task_table = list(zip(*([task_headers] + task_table)))
        print("\nTask Queue Information:")
        print(tabulate(transposed_task_table, headers="firstrow", tablefmt="grid"))

    def get_stc(self, relative_path="tasks.csv"):
        # Save statistics to CSV
        fieldnames = [attr for attr in vars(self.statistics[0]) if not callable(getattr(self.statistics[0], attr)) and not attr.startswith("__")]

        for i in self.statistics:
            print(i.__dict__)

        filename = os.path.join(os.getcwd(), relative_path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w+', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for task in self.statistics:
                writer.writerow({k: v for k, v in vars(task).items() if k in fieldnames})


    def envaluate_action(self, action):
        task = self.get_task(action=action)
        reward = self.calculate_reward(task=task)
        return reward

def make_env(training_num=1, test_num=1, state_dim = 1,task_num = 32,
             node_num=10, queue_len=10,
             co_num = [1,2,4], seed = 233, cp = None,
             min_task_arrival_rate=0.05,max_task_arrival_rate=0.09,T = 1,
             node_port = [16121,16121,16121,16121],wq = 250,wtt = 1,wt = 10,t = 1):
    if not cp:
        cp = [30]*node_num
    if isinstance(cp,int):
        cp = [cp]*node_num
    ENV = AIGCEnv(node_num=node_num,
                  queue_len=queue_len,
                  task_num=task_num,
                  cp = [30]*(int)(node_num/2)+[20]*(int)(node_num - node_num/2),
                  w_t=wt,w_q=wq,w_tt=wtt,load_time=5,
                  co_num = co_num,seed = seed,
                  state_dim=state_dim,
                  min_task_arrival_rate=min_task_arrival_rate,
                  max_task_arrival_rate=max_task_arrival_rate,T = T,t = t)

    def _select_env(evaluate = False):
        ENV.evaluate = evaluate
        return ENV

    env = _select_env()

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: _select_env() for _ in range(training_num)])

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: _select_env(True) for _ in range(test_num)])

    return env, train_envs, test_envs
