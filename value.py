class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma, nrow, ncol):
        self.env = env
        self.nrow = nrow
        self.ncol = ncol
        self.v = [0] * self.ncol * self.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.ncol * self.nrow)]

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.ncol * self.nrow
            for s in range(self.ncol * self.nrow):
                v_list = []
                for a in range(4):
                    p, next_state, reward, done = self.env.P[s][a][0]
                    v = reward + p * self.gamma * self.v[next_state] * (1-done)
                    v_list.append(v)
                new_v[s] = max(v_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
       
        print("价值迭代一共进行%d轮" % cnt)
        return self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略
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
        return self.pi