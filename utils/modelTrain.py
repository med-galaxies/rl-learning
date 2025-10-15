import matplotlib.pyplot as plt
import numpy as np
import torch, random, time
from tqdm import tqdm  
from algorithm.nStepSarsa import NStepSarsa
from algorithm.dynaQ import DynaQ
from algorithm.DQN import DQN
from algorithm.doubleDQN import DoubleDQN
from utils import rl_utils
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
                env_step = 0
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or env_step >= 500
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                    env_step += 1
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
                env_step = 0
                while not done:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or env_step >= 500
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    action = next_action
                    env_step += 1
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
        eposide_list = list(range(len(return_list)))
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
        eposide_list = list(range(len(return_list)))
        plt.plot(eposide_list, return_list, label=str(n_step)+' steps Sarsa')
    plt.legend()
    plt.xlabel('Eposides')
    plt.ylabel('Returns')
    plt.title('N-Step Sarsa on {}'.format('Cliff Walking'))
    plt.show()


def trainingDQN(env, env_name, num_episodes=500):
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("xpu") if torch.xpu.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    
    if "Pendulum" in env_name:
        action_dim = 11
    else:
        action_dim = env.action_space.n

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value) 
                    if "Pendulum" in env_name:
                        action_continuous = rl_utils.dis_to_con(action, env, agent.action_dim)
                        next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                        done = terminated or truncated or (env_step >= 200)
                    else:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated or (env_step >= 500)

                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    env_step += 1
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                        
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    rl_utils.show_loss(agent.loss_list)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN')
    plt.show()

    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


def trainingDoubleDQN(env, env_name, episodes_num=500):
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    update_freq = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("xpu") if torch.xpu.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_dim = 11
    else:
        action_dim = env.action_space.n

    agent = DoubleDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                      update_freq, device)
    return_list = []
    q_value_list = []
    max_q_value = 0
    for iteration in range(10):
        with tqdm(total=int(episodes_num / 10), desc='Iteration %d' % iteration) as pbar:
            for episode in range(int(episodes_num / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value) 
                    if "Pendulum" in env_name:
                        action_continuous = rl_utils.dis_to_con(action, env, agent.action_dim)
                        next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                        done = terminated or truncated or (env_step >= 200)
                    else:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated or (env_step >= 500)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    env_step += 1

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * iteration + episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    rl_utils.show_loss(agent.loss_list)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double DQN')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double DQN')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Double DQN on {}'.format(env_name))
    plt.show()


    