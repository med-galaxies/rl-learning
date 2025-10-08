import numpy as np
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state): 
        if type(state) is tuple:
            state = int(state[0])  # state是一个元组,第一个元素是状态值,第二个元素是额外信息
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action # 选取下一步的操作,具体实现为epsilon-贪婪

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        if type(s0) is tuple:
            s0 = int(s0[0])
        if type(s1) is tuple:
            s1 = int(s1[0])
        #print(f"update: s0:{s0}, a0:{a0}, r:{r}, s1:{s1}, a1:{a1}")
        #print(f"q_table: {self.Q_table[s0][a0]}, q_table2: {self.Q_table[s1][a1]}")
        self.Q_table[s0, a0] += self.alpha * (r + self.gamma * self.Q_table[s1, a1]
                                              - self.Q_table[s0, a0])