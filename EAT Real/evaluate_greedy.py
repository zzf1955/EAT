from _agent.random_agent import RandomAgent
from _agent.greedy_agent import GreedyAgent
from _SDEnv_real.env import make_env
from _SDEnv_real.test_env import test_env

seed = 42
evaluate_num = 4
node_num = 4
queue_len = 10
min_task_arrival_rate = 0.07
max_task_arrival_rate = 0.07
T = 1
co_num = [1,2,4]
task_num = 32
wq = 10
wtt = 1
wt = 250
diffusion_steps = 10

task_arrival_rate = min_task_arrival_rate


res_path = f"statistic/liner_{task_num}_{T}_{max_task_arrival_rate}_{min_task_arrival_rate}_{diffusion_steps}_{wq}_{wtt}_{wt}_{node_num}_{queue_len}_{co_num}_{seed}/greedy_res.csv"
data_path = f"statistic/liner_{task_num}_{T}_{max_task_arrival_rate}_{min_task_arrival_rate}_{diffusion_steps}_{wq}_{wtt}_{wt}_{node_num}_{queue_len}_{co_num}_{seed}/greedy_data.csv"


if __name__ == "__main__":

    env, train_envs, test_envs = make_env(1, 1,
                                            queue_len=queue_len,
                                            node_num=node_num,
                                            state_dim=2,
                                            co_num=co_num,
                                            task_num = task_num,
                                            wq = wq,wtt = wtt,wt = wt,
                                            max_task_arrival_rate = max_task_arrival_rate,
                                            min_task_arrival_rate = min_task_arrival_rate,
                                            T = T)

    env.evaluate=True

    agent = GreedyAgent(env.action_space.shape[0],-1,1)
    test_env(env = env,agent = agent,render=False,episode_num=4,res_path = res_path)
    env.get_stc(data_path)

