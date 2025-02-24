import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from diffusion import Diffusion
from diffusion.model import AttentionMLP
from _SDEnv.env import make_env

@dataclass
class TrainingConfig:
    node_num: int = 8
    queue_len: int = 10
    max_task_arrival_rate: float = 0.15
    min_task_arrival_rate: float = 0.15
    co_num: Tuple[int] = (1, 2, 4, 8)
    state_dim: int = 2
    timesteps: int = 2
    trait_dim: int = 256
    
    actor_hidden_dims: Tuple[int] = (256, 256)
    critic_hidden_dims: Tuple[int] = (256, 256)
    diffusion_steps: int = 10
    
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.005
    alpha: float = 0.05
    tau: float = 0.005
    gamma: float = 0.95
    n_step: int = 3
    
    buffer_size: int = 1000000
    epochs: int = 5000
    steps_per_epoch: int = 100
    batch_size: int = 512
    training_num: int = 1
    test_num: int = 1
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    model_filename: str = "adsac_policy.pth"
    log_root: Path = Path("experiments/adsac_diffusion")

def setup_environment(cfg):
    return make_env(
        training_num=cfg.training_num,
        test_num=cfg.test_num,
        queue_len=cfg.queue_len,
        node_num=cfg.node_num,
        co_num=cfg.co_num,
        seed=cfg.seed,
        state_dim=cfg.state_dim,
        max_task_arrival_rate=cfg.max_task_arrival_rate,
        min_task_arrival_rate=cfg.min_task_arrival_rate
    )

def create_actor_components(state_dim, action_dim, cfg):
    actor_net = AttentionMLP(
        state_dim=state_dim,
        action_dim=cfg.trait_dim,
        hidden_dim=cfg.actor_hidden_dims
    )
    diffusion = Diffusion(
        input_dim=state_dim,
        output_dim=cfg.trait_dim,
        model=actor_net,
        max_action=1.0,
        n_timesteps=cfg.diffusion_steps
    ).to(cfg.device)
    actor = ActorProb(
        preprocess_net=diffusion,
        action_shape=(action_dim,),
        unbounded=False,
        device=cfg.device,
        preprocess_net_output_dim=cfg.trait_dim
    ).to(cfg.device)
    optimizer = optim.Adam(
        actor.parameters(),
        lr=cfg.actor_lr,
        weight_decay=cfg.weight_decay
    )
    return actor, optimizer

def create_critic_components(state_dim, action_dim, cfg):
    def _build_critic():
        net = Net(
            state_dim,
            action_dim,
            hidden_sizes=cfg.critic_hidden_dims,
            activation=nn.Mish,
            concat=True,
            device=cfg.device
        )
        return Critic(net, device=cfg.device).to(cfg.device)
    
    critic1 = _build_critic()
    critic1_optim = optim.Adam(
        critic1.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.weight_decay
    )
    critic2 = _build_critic()
    critic2_optim = optim.Adam(
        critic2.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.weight_decay
    )
    return critic1, critic1_optim, critic2, critic2_optim

def setup_experiment(cfg):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cfg.log_root / f"{cfg.node_num}nodes" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def main(cfg):
    env, train_envs, test_envs = setup_environment(cfg)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log_dir = setup_experiment(cfg)
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    actor, actor_optim = create_actor_components(state_shape, action_shape, cfg)
    critic1, critic1_optim, critic2, critic2_optim = create_critic_components(state_shape, action_shape, cfg)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=cfg.tau,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        estimation_step=cfg.n_step,
    )

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(cfg.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=cfg.epochs,
        step_per_epoch=100,
        step_per_collect = None,
        episode_per_test=1,
        episode_per_collect=1,
        batch_size=cfg.batch_size,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), log_dir / cfg.model_filename),
        logger=logger,
        test_in_train=False
    )

    torch.save(policy.state_dict(), log_dir / f"final_{cfg.model_filename}")

if __name__ == "__main__":
    config = TrainingConfig()
    main(config)
