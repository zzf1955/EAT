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

module_name = "dsac_policy.pth"

actor_lr = 3e-4
critic_lr = 3e-4
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256,256]
diffusion_steps = 10

node_num = 8
queue_len = 10

max_task_arrival_rate = 0.15
min_task_arrival_rate = 0.15
T = 2

trait_dim = 256
co_num = [1,2,4,8]
seed = 0

alpha = 0.05
tau = 0.005

buffer_size = 1000000
epoch = 5000
step_per_epoch = 4
episode_per_collect = 1
episode_per_test = 1
repeat_per_collect = 1
update_per_step = 1
batch_size = 512
gamma = 0.95
n_step = 3
training_num = 1
test_num = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wd = 0.005



# Define actor and critic models separately
def create_actor(state_shape, action_shape):
    # Actor network
    actor_net = MLP(
        state_dim = state_shape,
        action_dim = trait_dim,
        hidden_dim = actor_hidden_dims
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

    # Create environment
    env, train_envs, test_envs = make_env(training_num, test_num, 
                                            queue_len=queue_len,
                                            node_num=node_num,
                                            co_num = co_num,
                                            seed = seed,
                                            state_dim=2,
                                            min_task_arrival_rate = min_task_arrival_rate,
                                            max_task_arrival_rate = max_task_arrival_rate,
                                            T = T)

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

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')

    logdir = f"liner_subpro_{node_num}nodes_wq=10/{T}_{max_task_arrival_rate}_{min_task_arrival_rate}_{node_num}_{queue_len}_{co_num}"
    log_path = os.path.join(
        logdir, 'dsac', str(time_now)
    )

    log_res_path = os.path.join(
        logdir, 'dsac_f', str(time_now)
    )

    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, module_name))

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
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(buffer_size, 1))
    test_collector = Collector(policy, test_envs)

    result = offpolicy_trainer(
        policy = policy,
        train_collector = train_collector,
        test_collector = test_collector,
        max_epoch = epoch,
        step_per_epoch = step_per_epoch,
        episode_per_test = episode_per_test,
        step_per_collect = None,
        batch_size = batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=update_per_step,
        episode_per_collect = episode_per_collect,
        test_in_train=False,
    )
    print(result)
    
    if not os.path.isdir(log_res_path):
        os.makedirs(log_res_path)
    try:
        torch.save(policy.state_dict(), os.path.join(log_res_path, module_name))
    except:
        torch.save(policy.state_dict(), os.path.join(log_path, module_name))

if __name__ == '__main__':
    main()
