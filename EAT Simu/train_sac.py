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

from _SDEnv.env import make_env

@dataclass
class TrainingConfig:
    node_num: int = 8
    queue_len: int = 10
    task_arrival_rate: float = 0.15
    co_num: Tuple[int] = (1, 2, 4, 8)
    state_dim: int = 2
    
    actor_hiddens: Tuple[int] = (256, 256)
    critic_hiddens: Tuple[int] = (256, 256)
    
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.005
    buffer_size: int = 1_000_000
    batch_size: int = 512
    gamma: float = 0.95
    tau: float = 0.005
    alpha: float = 0.05
    n_step: int = 3
    
    epochs: int = 5000
    steps_per_epoch: int = 100
    episodes_per_collect: int = 1
    seed: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_filename: str = "sac_policy.pth"
    log_root: Path = Path("experiments/sac_diffusion")

def setup_environment(cfg: TrainingConfig):
    return make_env(
        training_num=1,
        test_num=1,
        queue_len=cfg.queue_len,
        node_num=cfg.node_num,
        co_num=cfg.co_num,
        seed=cfg.seed,
        state_dim=cfg.state_dim,
        max_task_arrival_rate=cfg.task_arrival_rate,
        min_task_arrival_rate=cfg.task_arrival_rate
    )

def build_actor(state_dim: int, action_dim: int, cfg: TrainingConfig):
    net = Net(
        state_dim,
        hidden_sizes=cfg.actor_hiddens,
        activation=nn.Mish,
        device=cfg.device
    )
    actor = ActorProb(
        net,
        action_dim,
        device=cfg.device
    ).to(cfg.device)
    optimizer = optim.Adam(
        actor.parameters(),
        lr=cfg.actor_lr,
        weight_decay=cfg.weight_decay
    )
    return actor, optimizer

def build_critic(state_dim: int, action_dim: int, cfg: TrainingConfig):
    def _create_critic():
        net = Net(
            state_dim,
            action_dim,
            hidden_sizes=cfg.critic_hiddens,
            activation=nn.Mish,
            concat=True,
            device=cfg.device
        )
        return Critic(net, device=cfg.device).to(cfg.device)
    
    critic1 = _create_critic()
    critic1_optim = optim.Adam(
        critic1.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.weight_decay
    )
    
    critic2 = _create_critic()
    critic2_optim = optim.Adam(
        critic2.parameters(),
        lr=cfg.critic_lr,
        weight_decay=cfg.weight_decay
    )
    return critic1, critic1_optim, critic2, critic2_optim

def create_experiment_dir(cfg: TrainingConfig) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = cfg.log_root / f"{cfg.node_num}nodes" / timestamp
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
    critic1, critic1_optim, critic2, critic2_optim = build_critic(state_shape, action_shape, cfg)
    
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
    
    buffer = VectorReplayBuffer(cfg.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    
    save_path = log_dir / cfg.model_filename
    def save_best_fn(policy):
        torch.save(policy.state_dict(), save_path)
    
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
    print(f"Training completed. Final model saved to {final_model_path}")
    return result

if __name__ == "__main__":
    config = TrainingConfig(
        node_num=8,
        queue_len=10,
        task_arrival_rate=0.15,
        seed=0
    )
    
    main(config)
