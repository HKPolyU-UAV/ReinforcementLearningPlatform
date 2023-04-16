from common.common_cls import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
"""use CPU or GPU"""


class DQN:
    def __init__(self,
                 env,
                 gamma: float = 0.9,
                 epsilon: float = 0.95,
                 learning_rate: float = 0.01,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 target_replace_iter: int = 100,
                 eval_net: DQNNet = DQNNet(),
                 target_net: DQNNet = DQNNet()):
        """
        @param env:
        @param gamma:
        @param epsilon:
        @param learning_rate:
        @param memory_capacity:
        @param batch_size:
        @param target_replace_iter:
        @param eval_net:
        @param target_net:
        """
        '''DQN'''
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        self.episode = 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        '''DQN'''
        self.env = env
        '''From rl_base'''
        # DQN 要求智能体状态必须是连续的，但是动作必须是离散的
        # DQN 状态维度可以是很多，每增加一维，神经网络输出就增加一维
        # DQN 动作是离散的，所以神经网络输出维度等于动作空间中动作的数量。动作每增加一维，神经网络输出维度就增加该维对应的动作数量
        # agentName:            the name of the agent
        # state_dim_nn:         the dimension of the neural network input
        # action_dim_nn:        the dimension of the neural network output
        # action_dim_physical:  the dimension of the physical action
        # action_space:         action space with all physical action in it, PHYSICAL!!, NOT THE NUMBER OF THE ACTION!!
        '''From rl_base'''

        '''NN'''
        self.eval_net = eval_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.env.state_dim, self.env.action_dim)
        self.target_replace_count = 0
        '''NN'''

        '''datasave'''
        self.save_step = []         # step-TDError-NNLoss，存储 步数-TD误差-神经网络损失函数
        self.save_TDError = []      #
        self.save_NNLose = []       #
        self.save_episode = []      # episode-reward-epsilon，存储 回合-该回合累计奖励-探索概率
        self.save_reward = []       #
        self.save_epsilon = []      #
        '''datasave'''

    def get_action_random(self):
        """
        :brief:         choose an action randomly
        :return:        the number of the action
        """
        # random.seed()
        _a = []
        for _num in self.env.action_space:
            _a.append(np.random.choice(_num))
        return np.array(_a)

    def get_action_optimal_in_DQN(self, state):
        """
        :brief:         choose an action greedy
        :param state:   state
        :return:        the number of the action
        """
        t_state = torch.tensor(state).float().to(device)
        t_action_value = self.target_net(t_state).cpu().detach().numpy()
        # print(t_action_value.shape)
        # num = np.random.choice(np.where(t_action_value == np.max(t_action_value))[0])
        num = np.argmax(t_action_value)
        return num

    def get_action_with_fixed_epsilon(self, state, epsilon):
        """
        :brief:             choose an action with a certain exploration probability
        :param state:       state
        :param epsilon:     exploration probability
        :return:            the number of the action
        """
        # random.seed()
        self.epsilon = epsilon
        if random.uniform(0.0, 1.0) < self.epsilon:
            return self.get_action_random()
        else:
            return self.get_action_optimal_in_DQN(state)

    def actionNUm2PhysicalAction(self, action):
        """
        :brief:             Convert action number to physical action
        :param action:      the number if the action
        :return:            physical action
        :rules:             Action in higher-dimension has higher priority. For example, if the action space is
                            [['a', 'b', 'c'], ['d', 'e', 'f', 'g'], ['h', 'i', 'j', 'k', 'l']], then the dimension is 3,
                            and the length of each dimension is 3, 4, and 5. We will use some examples to illustrate
                            the relationship between actionNum and physicalAction:
                            actionNum   ->    physicalAction
                                0       ->    ['a', 'd', 'h']
                                1       ->    ['a', 'd', 'i']
                                5       ->    ['a', 'e', 'h']
                                8       ->    ['a', 'e', 'k']
                                20      ->    ['b', 'd', 'h']
        """
        actionSpace = self.env.action_space.copy()
        physicalAction = []
        count = 0
        for _item in reversed(actionSpace):       # 反序查找
            length = len(_item)
            index = action % length
            physicalAction.append(_item[index])
            count += 1
            action = int(action / length)
        physicalAction.reverse()
        return physicalAction

    def get_epsilon(self):
        """
        :brief:             get the exploration probability of an episode
        :return:            episode
        """
        # self.epsilon = 0.2       # It is a user-defined module.
        """Episode-Epsilon for Flight Attitude Simulator"""
        if 0 <= self.episode <= 300:
            self.epsilon = 2.222e-06 * self.episode ** 2 - 0.001667 * self.episode + 0.9     # FAS
        elif 300 < self.episode <= 600:
            self.epsilon = 2.222e-06 * self.episode ** 2 - 0.003 * self.episode + 1.45          # FAS
        elif 600 < self.episode <= 900:
            self.epsilon = 2.222e-06 * self.episode ** 2 - 0.004333 * self.episode + 2.4       # FAS
        elif 900 < self.episode <= 1200:
            self.epsilon = 2.222e-06 * self.episode ** 2 - 0.005667 * self.episode + 3.75       # FAS
        else:
            self.epsilon = 0.1
        """Episode-Epsilon for Flight Attitude Simulator"""

        """Episode-Epsilon for Nav Empty World"""
        if 0 <= self.episode <= 500:
            self.epsilon = -0.0006 * self.episode + 0.9
        elif 500 < self.episode <= 1000:
            self.epsilon = -0.0006 * self.episode + 1.05
        elif 1000 < self.episode <= 1500:
            self.epsilon = -0.0006 * self.episode + 1.2
        elif 1500 < self.episode <= 2000:
            self.epsilon = -0.0006 * self.episode + 1.35
        else:
            self.epsilon = 0.1
        """Episode-Epsilon for Nav Empty World"""
        return self.epsilon

    def torch_action2num(self, batch_action_number: np.ndarray):
        row = batch_action_number.shape[0]
        col = batch_action_number.shape[1]
        index_a = []
        for j in range(row):
            index = []
            for i in range(col):
                index.append(np.squeeze(np.argwhere(batch_action_number[j, i] == self.env.action_space[i])).tolist())
            index_a.append(index.copy())
            # print(index)
        # print(index_a)
        numpy_a = np.array(index_a)
        # print(numpy_a)
        res = []
        for i in range(row):
            temp = numpy_a[i]
            j = col - 1
            k = 1
            _sum = 0
            while j >= 0:
                _sum += k * temp[j]
                k *= self.env.action_num[j]
                j -= 1
            res.append(_sum)
        return torch.unsqueeze(torch.tensor(res).long(), dim=1)

    def nn_training(self, saveNNPath=None, is_reward_ascent=False):
        """
        :param is_reward_ascent:
        :brief:             train the neural network
        :param saveNNPath:  path of the pkl file
        :return:            None
        """
        self.target_replace_count += 1
        if self.target_replace_count % self.target_replace_iter == 0:       # 满足这个条件，网络参数就更新一次
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.target_net, saveNNPath + '/' + 'dqn.pkl')
            torch.save(self.target_net.state_dict(), saveNNPath + '/' + 'dqn_parameters.pkl')
            torch.save(self.eval_net, saveNNPath + '/' + 'eval_dqn.pkl')
            torch.save(self.eval_net.state_dict(), saveNNPath + '/' + 'eval_dqn_parameters.pkl')
            print('网络更新：', int(self.target_replace_count / self.target_replace_iter))
        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        # 按照奖励的升序排列，得到索引号
        t_s = torch.tensor(state, dtype=torch.float).to(device)
        t_a_pos = self.torch_action2num(action).to(device)  # t_a是具体的物理动作，需要转换成动作编号作为索引值，是个tensor
        t_r = torch.unsqueeze(torch.tensor(reward, dtype=torch.float).to(device), dim=1)
        t_s_ = torch.tensor(new_state, dtype=torch.float).to(device)
        t_bool = torch.unsqueeze(torch.tensor(done, dtype=torch.float).to(device), dim=1)
        q_next = self.target_net(t_s_).detach().to(device)
        res = torch.max(input=q_next, dim=1, keepdim=True)
        q_target = t_r + self.gamma * (res[0].mul(t_bool))
        for _ in range(1):
            q_eval = self.eval_net(t_s).gather(1, t_a_pos)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saveData_StepTDErrorNNLose(self.target_replace_count,
                                            (q_target - q_eval).sum().detach().cpu().numpy(),
                                            loss.detach().cpu().numpy())

    def get_optimalfrompkl(self, nn_para=None):
        """
        :brief:             加载最优控制器
        :param nn_para:     最优神经网络控制器的参数文件
        :return:            None
        """
        self.target_net.load_state_dict(torch.load(nn_para))

    def DQN_info(self):
        print('DQN agent name:', self.env.name)
        print('DQN input dimension:', self.env.state_dim)
        print('DQN output dimension:', self.env.action_num)
        print('Agent physical action dimension:', self.env.action_dim)
        print('Agent action space:', self.env.action_space)
        print('Replay memory capitaty:', self.memory_capacity)
        print('Batch size:', self.batch_size)

    def saveData_EpisodeRewardEpsilon(self, episode, reward, epsilon, is2file=False, filename='EpisodeRewardEpsilon.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'episode:': self.save_episode,
                'reward': self.save_reward,
                'epsilon': self.save_epsilon,
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_episode.append(episode)
            self.save_reward.append(reward)
            self.save_epsilon.append(epsilon)

    def saveData_StepTDErrorNNLose(self, step, tderror, nnlose, is2file=False, filename='StepTDErrorNNLose.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'Step:': self.save_step,
                'TDError': self.save_TDError,
                'NNLose': self.save_NNLose,
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_step.append(step)
            self.save_TDError.append(tderror)
            self.save_NNLose.append(nnlose)
