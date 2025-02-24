import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
from diffusion.model import MLP
from _SDEnv.env import make_env

@dataclass
class TrainingConfig:
    
    node_num: int = 8
    queue_len: int = 10
    co_numbers: Tuple[int] = (1, 2, 4, 8)
    max_task_rate: float = 0.15
    min_task_rate: float = 0.15
    timesteps: int = 2

    actor_hidden_dims: Tuple[int] = (256, 256)
    critic_hidden_dims: Tuple[int] = (256, 256)
    trait_dim: int = 256
    diffusion_steps: int = 10

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    epochs: int = 5000
    batch_size: int = 512
    buffer_size: int = 1_000_000
    alpha: float = 0.05
    tau: float = 0.005
    gamma: float = 0.95

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.005

    model_filename: str = "dsac_policy.pth"
    log_root: Path = Path("experiments/dsac_diffusion")

def build_actor(
    state_dim: int,
    action_dim: int,
    cfg: TrainingConfig
) -> Tuple[ActorProb, optim.Adam]:

    actor_net = MLP(
        state_dim=state_dim,
        action_dim=cfg.trait_dim,
        hidden_dim=cfg.actor_hidden_dims
    )

    diffusion_model = Diffusion(
        input_dim=state_dim,
        output_dim=cfg.trait_dim,
        model=actor_net,
        max_action=1.0,
        n_timesteps=cfg.diffusion_steps
    ).to(cfg.device)

    actor = ActorProb(
        preprocess_net=diffusion_model,
        action_shape=action_dim,
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

def build_critic(
    state_dim: int,
    action_dim: int,
    cfg: TrainingConfig
) -> Tuple[Critic, optim.Adam]:
    
    net = Net(
        state_dim,
        action_dim,
        hidden_sizes=cfg.critic_hidden_dims,
        activation=nn.Mish,
        concat=True,
        device=cfg.device
    )

    critic = Critic(net, device=cfg.device).to(cfg.device)

    
    optimizer = optim.Adam(
        critic.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.weight_decay
    )

    return critic, optimizer

def setup_environment(cfg: TrainingConfig) -> Tuple[object, object, object]:
    
    env, train_envs, test_envs = make_env(
        training_num=1,
        test_num=1,
        queue_len=cfg.queue_len,
        node_num=cfg.node_num,
        co_num=cfg.co_numbers,
        seed=cfg.seed,
        state_dim=2,
        min_task_arrival_rate=cfg.min_task_rate,
        max_task_arrival_rate=cfg.max_task_rate,
    )
    return env, train_envs, test_envs

def create_experiment_dir(cfg: TrainingConfig) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{cfg.timesteps}_{cfg.node_num}n_{cfg.queue_len}q"
    log_dir = cfg.log_root / exp_name / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, timestamp

def main(cfg: TrainingConfig):
    
    env, train_envs, test_envs = setup_environment(cfg)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    log_dir, timestamp = create_experiment_dir(cfg)
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    actor, actor_optim = build_actor(state_shape, action_shape, cfg)
    critic1, critic1_optim = build_critic(state_shape, action_shape, cfg)
    critic2, critic2_optim = build_critic(state_shape, action_shape, cfg)

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
        estimation_step=3,
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
        save_best_fn=lambda policy: torch.save(
            policy.state_dict(),
            log_dir / cfg.model_filename
        ),
        logger=logger,
        update_per_step=1,
        test_in_train=False
    )

    final_model_path = log_dir / f"final_{cfg.model_filename}"
    torch.save(policy.state_dict(), final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")

if __name__ == "__main__":
    
    train_cfg = TrainingConfig(
        node_num=8,
        queue_len=10,
        max_task_rate=0.15,
        min_task_rate=0.15,
        timesteps=2
    )

    main(train_cfg)

if __name__ == '__main__':
    main()
