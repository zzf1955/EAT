import os
import torch
import numpy as np
import argparse
import itertools
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic 
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from diffusion import Diffusion
from diffusion.model import MLP
from datetime import datetime
from _SDEnv.env import make_env
import pandas as pd

evaluate_num = 1
node_num = 4
queue_len = 10
task_arrival_rate = 0.14
co_num = [1,2,4]
time_limit = 200

module_path = f"res_policy/2_0.05_0.05_4_10_[1, 2, 4]/sac_policy.pth"
res_path = None
data_path = None


trait_dim = 256
actor_lr = 3e-4
critic_lr = 3e-4
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256,256]
diffusion_steps = 10


alpha = 0.05
tau = 0.005

seed = 0
buffer_size = 1000000
epoch = 15000
step_per_epoch = 100
episode_per_collect = 1
episode_per_test = 1
repeat_per_collect = 1
update_per_step = 1
batch_size = 512
gamma = 0.95
n_step = 3
training_num = 1
test_num = 1
logdir = ""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wd = 0.005

# Define actor and critic models separately
def create_actor(state_shape, action_shape):
    net = Net(
        state_shape,
        hidden_sizes=actor_hidden_dims,
        activation=nn.Mish,
        device=device
    )
    actor = ActorProb(
        net,
        action_shape,
        device=device
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=actor_lr,
        weight_decay=wd
    )
    return actor, actor_optim

def create_critic(state_shape, action_shape):
    # Critic networks
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(
        critic1.parameters(),
        lr=critic_lr,
        weight_decay=wd
    )

    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=device
    )
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(),
        lr=critic_lr,
        weight_decay=wd
    )

    return critic1, critic1_optim, critic2, critic2_optim

def main():

        # Create environment
    env, train_envs, test_envs = make_env(training_num, test_num, 
                                            queue_len=queue_len,
                                            node_num=node_num,
                                            task_arrival_rate=task_arrival_rate,
                                            state_dim=1,
                                            co_num=co_num)

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = state_shape[0]
    action_shape = action_shape[0]
    print("state_dims:",state_shape)
    print("action_dims:",action_shape)
    print("arrival_rate:",task_arrival_rate)

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create actor and critic networks
    actor, actor_optim = create_actor(state_shape, action_shape)
    critic1, critic1_optim, critic2, critic2_optim = create_critic(state_shape, action_shape)

    # Create policy
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
    )

    test_collector = Collector(policy, test_envs)

    print(f"Loading policy from {module_path}")
    policy.load_state_dict(torch.load(module_path, map_location=device))

    print("Evaluating the loaded policy...")
    test_envs.evaluate = True
    policy.eval()
    results = []
    for i in range(evaluate_num):
        test_collector.reset()
        np.random.seed(evaluate_num)
        torch.manual_seed(evaluate_num)
        result = test_collector.collect(n_episode=test_num, render=False)
        test_envs.workers[0].env.get_stc(data_path)
        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(res_path, index=False)

if __name__ == '__main__':
    for task_arrival_ratein in [0.01, 0.03 ,0.05, 0.07, 0.09, 0.11]:
        res_path = f"statistic/{node_num}/{task_arrival_rate}_{node_num}_{queue_len}_{co_num}_{time_limit}/sac_res.csv"
        data_path = f"statistic/{node_num}/{task_arrival_rate}_{node_num}_{queue_len}_{co_num}_{time_limit}/sac_data.csv"
        task_arrival_rate = task_arrival_rate
        main()