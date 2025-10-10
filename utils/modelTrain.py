import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  
import time
from algorithm.nStepSarsa import NStepSarsa
from algorithm.dynaQ import DynaQ
def trainingSarsa(env, agent, num_episodes=500):

    np.random.seed(0)
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()

def trainingQlearning(env, agent, num_episodes=500):

    np.random.seed(0)
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    return return_list

def trainingDynaQ(env, agent, num_episodes=500):
    np.random.seed(0)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_action = agent.take_action(next_state)
                    episode_return += reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


def trainingNStepSarsa(env, agent, num_episodes=500):
    np.random.seed(0)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_action = agent.take_action(next_state)
                    episode_return += reward
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list

def trainDynaQ(env, num_eposides=500):
    n_planning_list = [0, 2, 20]
    if hasattr(env, 'shape'):
        nrow, ncol = env.shape
    else:
        nrow, ncol = env.nrow, env.ncol
    for n_planning in n_planning_list:
        time.sleep(0.5)
        agent = DynaQ(ncol, nrow, 0.1, 0.1, 0.9, n_planning)
        return_list = trainingDynaQ(env, agent, num_eposides)
        eposide_list = list(range(return_list))
        plt.plot(eposide_list, return_list, label=str(n_planning)+' planning steps')
    plt.legend()
    plt.xlabel('Eposides')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()


def trainNStepSarsa(env, num_eposides=500):
    n_step_list = [0, 5, 10]
    if hasattr(env, 'shape'):
        nrow, ncol = env.shape
    else:
        nrow, ncol = env.nrow, env.ncol
    for n_step in n_step_list:
        time.sleep(0.5)
        agent = NStepSarsa(ncol, nrow, 0.1, 0.1, 0.9, 4, n_step)
        return_list = trainingNStepSarsa(env, agent, num_eposides)
        eposide_list = list(range(return_list))
        plt.plot(eposide_list, return_list, label=str(n_step)+' steps Sarsa')
    plt.legend()
    plt.xlabel('Eposides')
    plt.ylabel('Returns')
    plt.title('N-Step Sarsa on {}'.format('Cliff Walking'))
    plt.show()
    