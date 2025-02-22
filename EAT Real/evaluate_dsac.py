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
from _SDEnv_real.env import make_env
import pandas as pd

seed = 0
evaluate_num = 4
node_num = 4
queue_len = 10
min_task_arrival_rate = 0.07
max_task_arrival_rate = 0.07
T = 1
co_num = [1,2,4]
task_num = 32
wq = 10
wtt = 250
wt = 1
diffusion_steps = 10

module_path = f"upload_policy/2_0.05_0.09_4_10_[1, 2, 4]/dsac_policy_f.pth"
res_path = f"statistic/liner_{task_num}_{T}_{max_task_arrival_rate}_{min_task_arrival_rate}_{diffusion_steps}_{wq}_{wtt}_{wt}_{node_num}_{queue_len}_{co_num}_{seed}/dsac_res.csv"
data_path = f"statistic/liner_{task_num}_{T}_{max_task_arrival_rate}_{min_task_arrival_rate}_{diffusion_steps}_{wq}_{wtt}_{wt}_{node_num}_{queue_len}_{co_num}_{seed}/dsac_data.csv"

actor_lr = 3e-4
critic_lr = 3e-4
actor_hidden_dims = [256,256]
critic_hidden_dims = [256,256]
trait_dim = 256

alpha = 0.05
tau = 0.005

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

# Create environment
env, train_envs, test_envs = make_env(training_num, test_num,
                                        queue_len=queue_len,
                                        node_num=node_num,
                                        state_dim=2,
                                        co_num=co_num,
                                        task_num = task_num,
                                        wq = wq,wtt = wtt,wt = wt)

# Define actor and critic models separately
def create_actor(state_shape, action_shape):
    # Actor network
    actor_net = MLP(
        state_dim= state_shape,
        action_dim= trait_dim,
        hidden_dim= actor_hidden_dims
    )
    actor = Diffusion(
        input_dim=state_shape,
        output_dim=trait_dim,
        model = actor_net,
        max_action=1.,
        n_timesteps=diffusion_steps
    ).to(device)
    actor_prob = ActorProb(
        preprocess_net = actor,
        action_shape = action_shape,
        unbounded=False,
        device=device,
        preprocess_net_output_dim = trait_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor_prob.parameters(),
        lr=actor_lr,
        weight_decay=wd
    )
    return actor_prob, actor_optim


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
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    state_shape = state_shape[0]
    action_shape = action_shape[0]
    print("state_dims:",state_shape)
    print("action_dims:",action_shape)
    print("avg_arrival_rate:",(min_task_arrival_rate+max_task_arrival_rate)/2)
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
        result = test_collector.collect(n_episode=test_num, render=False)
        test_envs.workers[0].env.get_stc(data_path)
        results.append(result)
        print(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(res_path, index=False)

if __name__ == "__main__":
    main()
