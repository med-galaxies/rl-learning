import copy
import time
class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma, nrow, ncol):
        self.env = env
        self.nrow = nrow
        self.ncol = ncol
        self.v = [0] * self.ncol * self.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.ncol * self.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 计数器
        while True:
            new_v = [0] * self.ncol * self.nrow
            max_diff = 0
            for s in range(self.ncol * self.nrow):
                q_list = []
                for a in range(4):
                    q = 0
                    p, next_state, reward, done = self.env.P[s][a][0]
                    q += p*(reward + self.gamma * self.v[next_state] * (1-done))
                    q_list.append(q*self.pi[s][a])
                new_v[s] = sum(q_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.ncol * self.nrow):
            q_list = []
            
            for a in range(4):
                q = 0
                p, next_state, reward, done = self.env.P[s][a][0]
                q += p*(reward + self.gamma * self.v[next_state] * (1-done))
                q_list.append(q)
            max_q = max(q_list)
            max_q_cnt = q_list.count(max_q)
            self.pi[s] = [1/max_q_cnt if q == max_q else 0 for q in q_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break
        return self.pi, self.v

