import numpy as np
from _SDEnv.env import AIGCEnv

class GreedyAgent:
    def __init__(self, action_dim, high, low):
        self.action_dim = action_dim
        self.high = high
        self.low = low

    def act(self, env:AIGCEnv):
        max_draw_steps = env.max_draw_steps
        best_action = None
        best_reward = 0

        for steps in range(max_draw_steps):
            for task_id in range(min(env.task_queue.visible_len,len(env.task_queue.task_queue))):
                action = [steps/max_draw_steps]+[1]+[i == task_id for i in range(env.task_queue.visible_len)]
                action = np.array(action)*2-1
                reward = env.envaluate_action(np.array(action))
                if reward>best_reward:
                    best_reward = reward
                    best_action = action
        if best_reward == 0:
            return np.array([1,0,1]+[0]*(env.task_queue.visible_len-1))
        else:
            return np.array(best_action)


