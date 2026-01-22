import matplotlib.pyplot as plt
import numpy as np
import torch, random, time
from tqdm import tqdm  
from algorithm.n_step_sarsa import NStepSarsa
from algorithm.dyna_Q import DynaQ
from algorithm.DQN import DQN
from algorithm.double_DQN import DoubleDQN
from algorithm.double_DQN_noise import DoubleNoiseDQN
from algorithm.reinforce import Reinforce
from algorithm.actor_critic import ActorCritic
from algorithm.TRPO import TRPOContinuous as cTRPO
from algorithm.TRPO_fix import TRPO
from algorithm.PPO import PPO
from algorithm.ddpg import DDPG
from algorithm.SAC import SACContinuous as SAC
from algorithm.SAC_std import SACContinuous as SAC_std
from utils import rl_utils
from collections import deque
import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner

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


def trainingDQN(env, env_name, num_episodes=500, buffer_type='per'):
    lr = 2e-3
    num_episodes = 1000
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

    if buffer_type == 'per':
        replay_buffer = rl_utils.PrioritizedReplyBuffer(buffer_size)
    else:
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
                        if buffer_type == 'per':
                            b_s, b_a, b_r, b_ns, b_d, idxs, weights = replay_buffer.sample(batch_size, finish_ratio=i/10.0+0.1)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
                                'weights': weights
                            }
                            _, td_error = agent.update(transition_dict)
                            replay_buffer.update_priorities(idxs, np.abs(td_error))
                        else:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
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


def trainingDoubleDQN(env, env_name, episodes_num=500, buffer_type='per', isNoise=False):
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

    if buffer_type == 'per':
        replay_buffer = rl_utils.PrioritizedReplyBuffer(buffer_size)
    else:
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_dim = 11
    else:
        action_dim = env.action_space.n

    # agent = DoubleDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
    #                   update_freq, device)
    agent = DoubleNoiseDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                        update_freq, device, isNoise=isNoise)
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
                        if buffer_type == 'per':
                            b_s, b_a, b_r, b_ns, b_d, idxs, weights = replay_buffer.sample(batch_size, finish_ratio=iteration/10.0+0.1)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
                                'weights': weights
                            }
                            _, td_error = agent.update(transition_dict)
                            replay_buffer.update_priorities(idxs, np.abs(td_error))
                        else:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
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

def trainingReinforce(env, env_name, num_episodes=1000):
    learning_rate = 1e-3
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("xpu") if torch.cuda.is_available() else torch.device(
        "cpu")

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_dim = 11
    else:
        action_dim = env.action_space.n
    agent = Reinforce(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                    device)

    return_list = []
    q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    if "Pendulum" in env_name:
                        action_continuous = rl_utils.dis_to_con(action, env, action_dim)
                        next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                        done = terminated or truncated or (env_step >= 200)
                    else:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated or (env_step >= 500)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    env_step += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
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
    plt.title('REINFORCE')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()


def trainingActorCritic(env, env_name, num_episodes=1000):
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_dim = 11
    else:
        action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        gamma, device)

    return_list = []
    max_q_value = 0
    q_value_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                #print(f"initial state: {state}")
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value)
                    #print(f"before state: {state}")
                    if "Pendulum" in env_name:
                        action_continuous = rl_utils.dis_to_con(action, env, action_dim)
                        next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                        done = terminated or truncated or (env_step >= 200)
                    else:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated or (env_step >= 500)
                    #print(f"loop states: {state}")
                    if type(state) is tuple: 
                        transition_dict['states'].append(state[0])
                    else:
                        transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    env_step += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    rl_utils.show_loss(agent.actor_loss_list)
    rl_utils.show_loss(agent.critic_loss_list)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()



# ============= 8. 完整训练流程 =============
def trainingTRPO(env, action_type, env_name='CartPole-v1', num_episodes=500):
    """
    TRPO训练主循环
    
    Args:
        env_name: 环境名称
        max_episodes: 最大训练轮数
        max_steps: 每轮最大步数
        batch_size: 批量大小（轨迹长度）
    """
    # hidden_dim = 128
    # gamma = 0.9
    # lmbda = 0.9
    # critic_lr = 1e-2
    # kl_constraint = 0.00005
    # alpha = 0.5
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")

    num_episodes = 900
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    torch.manual_seed(0)
    # agent = TRPO(hidden_dim, env.observation_space, env.action_space,
    #                     lmbda, kl_constraint, alpha, critic_lr, gamma, device)


    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_type = 'Continuous'
        action_dim = env.action_space.shape[0]
    else:
        action_type = 'Discrete'
        action_dim = env.action_space.n

    agent = TRPO(state_dim, hidden_dim, action_dim, lmbda,
                kl_constraint, alpha, critic_lr, gamma, device, action_type)
    return_list = []
    q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value)

                        # action_continuous = rl_utils.dis_to_con(action, env, action_dim)
                        # next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                        

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or (env_step >= 500) if "Pendulum" in env_name \
                    else terminated or truncated or (env_step >= 400)

                    if type(state) is tuple: 
                        transition_dict['states'].append(state[0])
                    else:
                        transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    env_step += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    


    rl_utils.show_loss(agent.critic_loss_list)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

def trainingPPOOptuna(env, action_type, env_name, n_trials=50, n_episodes=100):
    """
    使用Optuna优化PPO超参数
    
    Args:
        env: gym环境
        action_type: 'Discrete' 或 'Continuous'
        env_name: 环境名称
        n_trials: Optuna试验次数
        n_episodes: 每次试验的训练episode数
    
    Returns:
        study: Optuna study对象
        best_params: 最佳参数字典
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    def objective(trial):
        # 定义超参数搜索空间
        actor_lr = trial.suggest_float('actor_lr', 1e-5, 5e-3, log=True)
        critic_lr = trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        gamma = trial.suggest_float('gamma', 0.95, 0.995)
        lmbda = trial.suggest_float('lmbda', 0.90, 0.99)
        epochs = trial.suggest_int('epochs', 3, 15)
        eps = trial.suggest_float('eps', 0.1, 0.3)
        
        # 约束: critic_lr应该 >= actor_lr
        if critic_lr < actor_lr:
            critic_lr = actor_lr * trial.suggest_float('lr_ratio', 1.0, 3.0)
        
        # 设备配置
        device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")
        
        # 随机种子
        torch.manual_seed(trial.number)
        np.random.seed(trial.number)
        random.seed(trial.number)
        
        # 环境配置
        state_dim = env.observation_space.shape[0]
        if "Pendulum" in env_name or action_type == 'Continuous':
            action_type_final = 'Continuous'
            action_dim = env.action_space.shape[0]
        else:
            action_type_final = 'Discrete'
            action_dim = env.action_space.n
        
        # 初始化agent
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, 
                   device, action_type_final, gamma, lmbda, epochs, eps)
        
        return_list = []
        
        # 训练循环
        for episode in range(n_episodes):
            episode_return = 0
            transition_dict = {
                'states': [], 
                'actions': [], 
                'next_states': [], 
                'rewards': [], 
                'dones': []
            }
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            env_step = 0
            
            # 单个episode采样
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # 根据环境设置done条件
                if "Pendulum" in env_name:
                    done = terminated or truncated or (env_step >= 500)
                else:
                    done = terminated or truncated or (env_step >= 400)
                
                if isinstance(state, tuple):
                    transition_dict['states'].append(state[0])
                else:
                    transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                state = next_state
                episode_return += reward
                env_step += 1
            
            return_list.append(episode_return)
            agent.update(transition_dict)
            
            # 中期报告（用于剪枝）
            if episode % 10 == 0 and episode > 0:
                intermediate_score = np.mean(return_list[-10:])
                trial.report(intermediate_score, episode)
                
                # 如果表现明显差于其他试验，提前终止
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # 返回最后20个episode的平均回报
        final_score = np.mean(return_list[-20:])
        return final_score
    
    # 创建study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # 优化
    print(f"\n开始优化PPO超参数 (环境: {env_name}, 动作类型: {action_type})")
    print(f"试验次数: {n_trials}, 每次episode数: {n_episodes}")
    print("=" * 60)
    
    study.optimize(objective, n_trials=n_trials, timeout=None,
                   show_progress_bar=True, n_jobs=1)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"最佳试验编号: {study.best_trial.number}")
    print(f"最佳平均回报: {study.best_value:.2f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        if key != 'lr_ratio':  # 跳过辅助参数
            print(f"  {key}: {value}")
    
    # 参数统计分析
    print("\n参数统计分析:")
    df = study.trials_dataframe()
    for param in ['actor_lr', 'critic_lr', 'hidden_dim', 'gamma', 'lmbda', 'epochs', 'eps']:
        if param in df.columns:
            param_col = f'params_{param}'
            if param_col in df.columns:
                print(f"  {param}:")
                print(f"    - 最优值: {study.best_params.get(param, 'N/A')}")
                print(f"    - 平均值: {df[param_col].mean():.4f}")
                print(f"    - 标准差: {df[param_col].std():.4f}")
    
    # 可视化
    try:
        import optuna.visualization as vis
        
        # 优化历史
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(f'ppo_{env_name}_optimization_history.html')
        
        # 参数重要性
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(f'ppo_{env_name}_param_importances.html')
        
        # 参数关系（平行坐标图）
        fig3 = vis.plot_parallel_coordinate(study, params=['actor_lr', 'critic_lr', 
                                                           'hidden_dim', 'gamma', 
                                                           'lmbda', 'epochs', 'eps'])
        fig3.write_html(f'ppo_{env_name}_parallel_coordinate.html')
        
        # 切片图（显示参数间的交互）
        fig4 = vis.plot_slice(study, params=['actor_lr', 'critic_lr', 'eps', 'epochs'])
        fig4.write_html(f'ppo_{env_name}_slice_plot.html')
        
        print(f"\n可视化已保存为HTML文件 (前缀: ppo_{env_name}_*.html)")
    except Exception as e:
        print(f"\n可视化生成失败: {e}")
    
    return study, study.best_params


def trainingPPO(env, action_type, env_name='CartPole-v1', num_episodes=500):
    actor_lr = 3e-4
    critic_lr = 5e-4
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    if "Pendulum" in env_name:
        action_type = 'Continuous'
        action_dim = env.action_space.shape[0]
    else:
        action_type = 'Discrete'
        action_dim = env.action_space.n

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, \
                action_type, gamma, lmbda, epochs, eps)
    return_list = []
    q_value_list = []
    max_q_value = 0
    for i in range(12):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value)

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or (env_step >= 500) if "Pendulum" in env_name \
                    else terminated or truncated or (env_step >= 400)

                    if type(state) is tuple: 
                        transition_dict['states'].append(state[0])
                    else:
                        transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    env_step += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)


    rl_utils.show_loss(agent.critic_loss_list)
    rl_utils.show_loss(agent.actor_loss_list)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

def trainTRPO(env, env_name):
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    torch.manual_seed(0)
    agent = cTRPO(hidden_dim, env.observation_space, env.action_space,
                        lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()





def trainingDDPG(env, env_name, episodes_num=2000, buffer_type='per', isNoise=False):
    # actor_lr = 3e-4
    # critic_lr = 5e-4
    # num_episodes = 200
    # hidden_dim = 64
    # gamma = 0.98
    # tau = 0.005  # 软更新参数
    # buffer_size = 10000
    # minimal_size = 1000
    # batch_size = 64
    # sigma = 0.01  # 高斯噪声标准差

    # actor_lr = 0.0005595877381331775
    # critic_lr = 0.0005670746943267102
    # num_episodes = 200
    # hidden_dim = 128
    # gamma = 0.9873801585815166
    # tau = 0.004511407069695454  # 软更新参数
    # buffer_size = 40000
    # minimal_size = 1000
    # batch_size = 128
    # sigma = 0.20706481081278097

    actor_lr = 7.88e-4
    critic_lr = 4.3e-4
    hidden_dim = 64
    gamma = 0.98884
    tau = 0.0035
    sigma = 0.2094
    batch_size = 128
    buffer_size = 40000
    minimal_size = 1000
    num_episodes = 200
    max_step = 1000

    num_iterations = 20
    device = torch.device("mps") if torch.mps.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

 
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0] 

    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, gamma, sigma, tau, device)

    return_list = []
    q_value_list = []
    max_q_value = 0
    for iteration in range(num_iterations):
        with tqdm(total=int(episodes_num / num_iterations), desc='Iteration %d' % iteration) as pbar:
            for episode in range(int(episodes_num / num_iterations)):
                episode_return = 0
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state, action) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value) 
            
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or (env_step >= max_step)
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
                            'dones': b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode+1) % num_iterations == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / num_iterations * iteration + episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-int(episodes_num / num_iterations):])
                    })
                pbar.update(1)
    rl_utils.show_loss(agent.actor_loss_list)
    rl_utils.show_loss(agent.critic_loss_list)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()


def trainingDDPGOptuna(env, env_name, episodes_num=100, n_trials=50):

    def objective(trial):
        # 定义超参数搜索空间
        actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
        critic_lr = trial.suggest_float("critic_lr", 1e-4, 2e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        gamma = trial.suggest_float("gamma", 0.95, 0.995)
        tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
        sigma = trial.suggest_float("sigma", 0.01, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        buffer_size = trial.suggest_int('buffer_size', 10000, 100000, step=10000)

        if critic_lr < actor_lr:
            critic_lr = actor_lr * trial.suggest_float('lr_ratio', 1.0, 3.0)

        device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
        
        # 随机种子
        random.seed(trial.number)
        np.random.seed(trial.number)
        torch.manual_seed(trial.number)
        
        # 初始化
        minimal_size = min(1000, buffer_size // 10)
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high[0]
        
        agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, 
                    actor_lr, critic_lr, gamma, sigma, tau, device)
        return_list = []

        for episode in range(episodes_num):
            episode_return = 0
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            env_step = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated or env_step >= 500
                replay_buffer.add(state, action, reward, next_state, terminated)
                state = next_state
                episode_return += reward
                env_step += 1
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)
            return_list.append(episode_return)

            if (episode + 1) % 10 == 0 and episode > 0:
                intermediate_score = np.mean(return_list[-10:])
                trial.report(intermediate_score, episode)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        final_score = np.mean(return_list[-10:])
        return final_score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True, n_jobs=1)

    # 输出结果
    print("\n" + "="*50)
    print("优化完成!")
    print("="*50)
    print(f"最佳试验编号: {study.best_trial.number}")
    print(f"最佳平均回报: {study.best_value:.2f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    try:
        import optuna.visualization as vis
        
        # 优化历史
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html('ddpg_optimization_history.html')
        
        # 参数重要性
        fig2 = vis.plot_param_importances(study)
        fig2.write_html('ddpg_param_importances.html')
        
        # 参数关系
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html('ddpg_parallel_coordinate.html')
        
        print("\n可视化已保存为HTML文件")
    except Exception as e:
        print(f"\n可视化生成失败: {e}")
    
    print(study.trials_dataframe().sort_values('value', ascending=False).head(10))
    return study, study.best_params


def trainingSAC(env, env_name, episodes_num=200, buffer_type='normal', isNoise=False):
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.001  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0] * 0.5
    max_step = 500

    num_iterations = 10
    device = torch.device("mps") if torch.mps.is_available() else torch.device(
        "cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

 
    if buffer_type == "per":
        replay_buffer = rl_utils.PrioritizedReplyBuffer(buffer_size)
    else:
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0] 

    # agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, 
    #             gamma, tau, device)
    agent = SAC_std(
        state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy,
        gamma, tau, device
    )

    return_list = []
    q_value_list = []
    max_q_value = 0
    for iteration in range(num_iterations):
        with tqdm(total=int(episodes_num / num_iterations), desc='Iteration %d' % iteration) as pbar:
            for episode in range(int(episodes_num / num_iterations)):
                episode_return = 0
                state = env.reset()
                done = False
                env_step = 0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state, action) * 0.005 + max_q_value * 0.995  # 平滑处理
                    q_value_list.append(max_q_value) 
            
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated or (env_step >= max_step)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    env_step += 1

                    if replay_buffer.size() > minimal_size:
                        if buffer_type == 'per':
                            b_s, b_a, b_r, b_ns, b_d, idxs, weights = replay_buffer.sample(batch_size, finish_ratio=iteration/10.0+0.1)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
                                'weights': weights
                            }
                            _, td_error = agent.update(transition_dict)
                            replay_buffer.update_priorities(idxs, np.abs(td_error))
                        else:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d,
                            }
                            agent.update(transition_dict)
                return_list.append(episode_return)
                if (episode+1) % num_iterations == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / num_iterations * iteration + episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-int(episodes_num / num_iterations):])
                    })
                pbar.update(1)
    # rl_utils.show_loss(agent.loss_list1)
    # rl_utils.show_loss(agent.loss_list2)
    # rl_utils.show_loss(agent.loss_list_alpha)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC')
    plt.show()


    frames_list = list(range(len(q_value_list)))
    plt.plot(frames_list, q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('SAC on {}'.format(env_name))
    plt.show()