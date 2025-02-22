import numpy as np

class RandomAgent:
    def __init__(self, action_dim, high, low):
        self.action_dim = action_dim
        self.high = high
        self.low = low

    def act(self, env):
        # 生成 action_dim 个随机数，在 [low, high) 区间内
        return np.random.uniform(self.low, self.high, self.action_dim)
