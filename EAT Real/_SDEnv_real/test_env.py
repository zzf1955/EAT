import pandas as pd

def test_env(env, agent, episode_num = 10, render = True,res_path = "random.csv"):

    episode_count = 0
    seed = 1
    avg_reward = 0
    results = []

    while True:
        seed+=1
        # 重置环境，开始新一轮
        observation = env.reset(seed)
        if render:
            print(f"Episode {episode_count + 1}: Reset")
            print(f"{observation}")
        
        done = False
        truncated = False
        total_reward = 0  # 当前轮的总奖励
        length = 0  # 当前轮的步数

        while not done and not truncated:
            # Agent 选择动作
            action = agent.act(env)

            # 执行动作
            observation, reward, done, truncated, info = env.step(action)

            if render:
                print(f"Step {length + 1}")
                print(f"Act : {action}")
                print(f"reward: {reward}")
            # 打印当前的 `observation` 和 `info`
            if render:
                env.render()

            # 累计奖励和步数
            total_reward += reward
            

            length += 1
            if render:
                input()

        # 打印每一轮的总奖励和步数
        print(f"Episode {episode_count + 1} Complete: Total Reward = {total_reward}, Length = {length}\n")
        avg_reward += total_reward
        # 记录轮数
        episode_count += 1
        results.append({"result":total_reward})

        # 选择继续或结束条件
        if episode_count >= episode_num: 
            break
    avg_reward/=episode_num
    print(f"avg re:{avg_reward}")
    results.append({"result":avg_reward})
    results_df = pd.DataFrame(results)
    results_df.to_csv(res_path, index=False)
    env.reset(233)

