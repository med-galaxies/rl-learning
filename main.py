from envs.env import CliffWalkingEnv
from algorithm.policyIteration import PolicyIteration
from algorithm.valueIteration import ValueIteration
from algorithm.sarsa import Sarsa
from utils.modelTrain import trainingSarsa, trainingQlearning, trainDynaQ, trainNStepSarsa, trainingDQN
from algorithm.qLearning import QLearning
import gymnasium as gym
import time

# def print_agent(agent, action_meaning, disaster=[], end=[], nrow=1, ncol=2):
#     print("状态价值：")
#     for i in range(nrow):
#         for j in range(ncol):
#             # 为了输出美观,保持输出6个字符
#             print('%6.6s' % ('%.3f' % agent.v[i * ncol + j]), end=' ')
#         print()

#     print("策略：")
#     for i in range(nrow):
#         for j in range(ncol):
#             # 一些特殊的状态,例如悬崖漫步中的悬崖
#             if (i * ncol + j) in disaster:
#                 print('****', end=' ')
#             elif (i * ncol + j) in end:  # 目标状态
#                 print('EEEE', end=' ')
#             else:
#                 a = agent.pi[i * ncol + j]
#                 pi_str = ''
#                 for k in range(len(action_meaning)):
#                     pi_str += action_meaning[k] if a[k] > 0 else 'o'
#                 print(pi_str, end=' ')
#         print()

def print_agent(agent, action_meaning, nrow, ncol, disaster=[], end=[]):
    for i in range(nrow):
        for j in range(ncol):
            if (i * ncol + j) in disaster:
                print('****', end=' ')
            elif (i * ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



def render_pic(agent, optimal_pi):
    state, _ = agent.env.reset()
    agent.env.render()
    # print(f"pi:{optimal_pi}, state:{state}")
    for step in range(50):
        action = optimal_pi[state].index(max(optimal_pi[state]))
        state, reward, terminated, truncated, _ = agent.env.step(action)
        agent.env.render()
        time.sleep(0.5)
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps, reward = {reward}")
            break
    
def modify_hole_reward(env, nrow, ncol, hole_penalty=-1.0):
    """
    修改冰湖环境中冰窟(H)的奖励为负值
    """
    for s in range(nrow * ncol):
        for a in range(4):
            for idx, (p, next_state, reward, done) in enumerate(env.P[s][a]):
                # 检查下一个状态是否是冰窟（H）
                y, x = divmod(next_state, ncol)  # 转成(row, col)
                if bytes(env.desc[y][x]) == b'H' and done:
                    env.P[s][a][idx] = (p, next_state, hole_penalty, done)
    return env


def main():
    #env = CliffWalkingEnv()
    # env = gym.make(
    #     'CliffWalking-v0',
    #     #'FrozenLake-v1',
    #     #map_name="4x4",
    #     is_slippery=False,
    #     render_mode="human"
    # )
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    env.reset()
    env.render()
    action_meaning = ['^', '>', 'v', '<']
    theta = 0.001
    gamma = 0.9
    num_eposide=500

    # if hasattr(env, 'shape'):
    #     nrow, ncol = env.shape
    # else:
    #     nrow, ncol = env.nrow, env.ncol
    
    #env = modify_hole_reward(env, nrow, ncol, -0.6)
    # agent = PolicyIteration(env, theta, gamma, nrow, ncol)
    # optimal_pi, optimal_v = agent.policy_iteration()
    # print_agent(agent, action_meaning, list(range(37, 47)), [47], nrow, ncol)
    # render_pic(agent, optimal_pi)

    # agent2 = ValueIteration(env, theta, gamma, nrow, ncol)
    # optimal_pi2 = agent2.value_iteration()
    # print_agent(agent2, action_meaning, list(range(37, 47)), [47], nrow, ncol)
    # render_pic(agent2, optimal_pi2)

    # agent = Sarsa(nrow, ncol, 0.1, 0.1, 0.9)
    # agent2 = QLearning(ncol, nrow, 0.1, 0.1, 0.9)
    # trainingSarsa(env,agent, num_eposide)
    # trainingQlearning(env, agent2, num_eposide)
    # #action_meaning = ['^', 'v', '<', '>']
    # print("Sarsa收敛策略:")
    # print_agent(agent, action_meaning, nrow, ncol, list(range(37, 47)), [47])
    # print("Q-Learning收敛策略:")
    # print_agent(agent2, action_meaning, nrow, ncol, list(range(37, 47)), [47])

    #trainNStepSarsa(env)
    #trainDynaQ(env)
    trainingDQN(env)
    env.close()

if __name__ == "__main__":
    main()
