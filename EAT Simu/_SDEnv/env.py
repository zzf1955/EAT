from _SDEnv.cluster import Cluster, Node
from _SDEnv.task import Task, TaskQueue, TaskGenerator
from tianshou.env import DummyVectorEnv
from tabulate import tabulate

import numpy as np
import os
import csv
import math
import random
import gymnasium as gym
from gymnasium import spaces

class AIGCEnv(gym.Env):
    def __init__(self, 
                queue_len, 
                task_arrival_rate, 
                node_num,
                cp:list[float],
                load_time=10,
                max_draw_steps=32,
                co_num=[1, 2, 4, 8], 
                w_t=3,
                w_q=10,
                w_tt = 1.2,
                task_num=32,
                time_limite=1024,
                steps_limite=100000,
                fixed_co=True,
                fixed_steps=False,
                fixed_size = True,
                seed = 233,
                state_dim = 1,
                min_task_arrival_rate = 0.0,
                max_task_arrival_rate = 1.0,
                T = 2,t = 0.1):
        super(AIGCEnv, self).__init__()

        self.set_seed(seed=seed)
        self.state_dim = state_dim

        # reward
        self.wt = w_t
        self.wq = w_q
        self.wtt = w_tt
        self.t = t

        # limite
        self.task_num = task_num
        self.time_limite = time_limite
        self.steps_limite = steps_limite

        self.min_task_arrival_rate = min_task_arrival_rate
        self.max_task_arrival_rate = max_task_arrival_rate
        self.task_arrival_rate = min_task_arrival_rate  # Initial task arrival rate
        self.T = T

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
            state_dim = self.state_dim)
        
        self.max_draw_steps = max_draw_steps
        self.current_time = 1
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + queue_len,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(queue_len*Task.vector_len + node_num*Node.vector_len,), dtype=np.float32)
        self.statistics = []

    def reset(self, SEED = 233):

        self.set_seed(seed=SEED)
        self.task_generator.reset()
        self.cluster.reset()
        
        self.task_queue = TaskQueue(
            self.task_generator, 
            visible_len=self.task_queue.visible_len, 
            init_job_num=self.task_queue.visible_len,
            state_dim=self.state_dim)
        
        self.current_time = 1
        self.current_step = 0
    
        return self.get_obs()

    def add_statistic(self, task: Task):
        steps = task.steps
        if steps < 5:
            quality = 0
        else:
            a = 0.16
            b = -0.02
            base = 6
            quality = a * np.log(steps) / np.log(base) + b
        if steps > 30:
            quality = 0.27
        quality_reward = self.wq * quality

        task.quality = quality
        task.waiting_time = self.current_time - task.arrival_time
        task.running_time =  task.duration
        task.response_time = task.waiting_time + task.running_time
        task.quality_reward = quality_reward

    def step(self, action):
        self.current_step += 1
        truncated = False
        done = False
        task = self.get_task(action)

        if (not task.valid) or (not task.execute):
            self.update_time(self.t)
        else:
            task.start_time = self.current_time
            self.task_queue.remove_task(task)
            self.cluster.launch_task(task=task)
            self.add_statistic(task=task)
            self.statistics.append(task)

        if self.current_time>self.time_limite or (self.current_step>self.steps_limite) :
            truncated = True
        if(self.task_generator.task_cnt>=self.task_num and self.task_queue.empty() and self.cluster.all_done()):
            done = True

        reward = self.calculate_reward(task=task)
        
        obs = self.get_obs()
        return obs, reward, done, truncated, {}

    def get_obs(self):
        if self.state_dim == 1:
            return np.hstack([self.task_queue.vector,self.cluster.vector])
        if self.state_dim == 2:
            return np.vstack([self.task_queue.vector,self.cluster.vector])

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
            task.steps = step_action*self.max_draw_steps

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
        return task

    def calculate_reward(self, task:Task):

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
            quality = 0.27

        time_reward = self.wt / (
            self.wtt * avg_waiting_time +
            (task.duration + self.current_time - task.arrival_time)
        )
        time_reward =min(10,time_reward)
        quality_reward = self.wq * quality

        return time_reward + quality_reward

    def update_time(self, working_time):
        self.cluster.working(working_time)
        if not self.task_generator.task_cnt>=self.task_num and self.current_time<self.time_limite:
            for i in range(int(working_time*10)):
                if random.random()>self.task_arrival_rate/10:
                    continue
                task = self.task_generator.get_new_task()
                task.arrival_time = self.current_time + i/10
                self.task_queue.task_queue.append(task)
        self.current_time += working_time

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

        transposed_node_table = list(zip(*([node_headers] + node_table)))
        print("Cluster Nodes Information:")
        print(tabulate(transposed_node_table, headers="firstrow", tablefmt="grid"))

        task_headers = ["Task ID", "Co-Number", "Arrival Time"]
        task_table = [
            [task.task_id, task.co_num, task.arrival_time]
            for task in self.task_queue.task_queue
        ]

        transposed_task_table = list(zip(*([task_headers] + task_table)))
        print("\nTask Queue Information:")
        print(tabulate(transposed_task_table, headers="firstrow", tablefmt="grid"))
        input()

    def get_stc(self, relative_path="tasks.csv"):
        # Save statistics to CSV
        fieldnames = [attr for attr in vars(self.statistics[0]) if not callable(getattr(self.statistics[0], attr)) and not attr.startswith("__")]

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

def make_env(training_num=1, test_num=1, state_dim = 1,
             node_num=10, queue_len=10, task_arrival_rate=0.5,
             co_num = [1,2,4,8], seed = 233, cp = None, time_limite = 1024,max_draw_steps  =32,t = 0.1):
    if not cp:
        cp = [30]*node_num
    if isinstance(cp,int):
        cp = [cp]*node_num
    ENV = AIGCEnv(node_num=node_num,
                  queue_len=queue_len,
                  task_arrival_rate=task_arrival_rate,
                  cp = [30] * node_num,max_draw_steps = max_draw_steps,
                  w_t=250,w_q=10,w_tt=1,load_time=5,
                  co_num = co_num,seed = seed,min_task_arrival_rate=task_arrival_rate,max_task_arrival_rate=task_arrival_rate,
                  state_dim=state_dim,time_limite=time_limite,t = t)

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