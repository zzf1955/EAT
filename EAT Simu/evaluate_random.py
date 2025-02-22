from _agent.random_agent import RandomAgent
from _agent.greedy_agent import GreedyAgent
from _SDEnv.env import make_env
from _SDEnv.test_env import test_env

node_num = 4
queue_len = 10
task_arrival_rate = 0.04
co_num = [1,2,4]
time_limit = 200
t = 1

res_path = None
data_path = None

if __name__ == "__main__":
    for task_arrival_rate in [0.01, 0.03 ,0.05, 0.07, 0.09, 0.11]:
        res_path = f"statistic/{node_num}/{task_arrival_rate}_{node_num}_{queue_len}_{co_num}_{time_limit}/random_res.csv"
        data_path = f"statistic/{node_num}/{task_arrival_rate}_{node_num}_{queue_len}_{co_num}_{time_limit}/random_data.csv"
        env, train_envs, test_envs = make_env(1, 1, 
                                            queue_len=queue_len,
                                            node_num=node_num,
                                            task_arrival_rate=task_arrival_rate,
                                            state_dim=2,max_draw_steps=32,
                                            co_num=co_num,t = t)
        env.evaluate=True
        agent = RandomAgent(env.action_space.shape[0],-1,1)
        test_env(env = env,agent = agent,render=False,episode_num=8,res_path=res_path,data_path = data_path)

print()