from _agent.random_agent import RandomAgent
from _agent.greedy_agent import GreedyAgent
from _SDEnv_real.env import make_env
from _SDEnv_real.test_env import test_env

if __name__ == "__main__":
    env = make_env(1,1,node_num=4,task_arrival_rate=0.4)[0]
    env.evaluate=True
    agent = RandomAgent(env.action_space.shape[0],-1,1)
    test_env(env = env,agent = agent,render=True,episode_num=64)

print()