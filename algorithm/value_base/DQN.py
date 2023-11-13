import math
import numpy as np
import torch
import torch.nn as nn
import os, time

from utils.classes import DQNNet, ReplayBuffer

"""use CPU or GPU"""
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
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
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
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
        # _a = []
        # for _num in self.env.action_num:
        #     _a.append(np.random.choice(_num))
        return np.random.randint(np.prod(self.env.action_num))

    def get_action_optimal_in_DQN(self, state):
        """
        :brief:         choose an action greedy
        :param state:   state
        :return:        the number of the action
        """
        t_state = torch.tensor(state).float().to(device)
        t_action_value = self.target_net(t_state).cpu().detach().numpy()
        return np.argmax(t_action_value)

    def get_action_with_fixed_epsilon(self, state, epsilon):
        """
        :brief:             choose an action with a certain exploration probability
        :param state:       state
        :param epsilon:     exploration probability
        :return:            the number of the action
        """
        # random.seed()
        self.epsilon = epsilon
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            return self.get_action_random()
        else:
            return self.get_action_optimal_in_DQN(state)

    def actionNUm2PhysicalAction(self, action):
        """
        :brief:             Convert action number to physical action
        :param action:      the number of the action
        :return:            physical action
        """
        # linear_action = []
        # for _a, _action_space in zip(action, self.env.action_space):
        #     linear_action.append(_action_space[_a])
        # return np.array(linear_action)
        actionSpace = self.env.action_space.copy()
        physicalAction = []
        count = 0
        for _item in reversed(actionSpace):  # 反序查找
            length = len(_item)
            index = math.floor(action % length)
            # print("_item:", _item, "index:", index)
            physicalAction.append(_item[index])
            count += 1
            action = int(action / length)
        physicalAction.reverse()
        return np.array(physicalAction)

    def torch_action2num(self, batch_action_number: np.ndarray):
        row = batch_action_number.shape[0]
        res = []
        for i in range(row):
            lastLen = 1
            actionIdx = 0
            idx = len(batch_action_number[i]) - 1
            for actions in reversed(self.env.action_space):
                actionIdx += actions.index(batch_action_number[i][idx]) * lastLen
                lastLen *= len(actions)
                idx -= 1
            res.append(actionIdx)
        return torch.tensor(res)

    def learn(self, path=None, is_reward_ascent=True, iter=10):
        self.target_replace_count += 1
        if self.target_replace_count % self.target_replace_iter == 0:       # 满足这个条件，网络参数就更新一次
            temp = path + 'trainNum_{}/'.format(self.target_replace_count)
            os.mkdir(temp)
            time.sleep(0.01)
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('Save net', self.target_replace_count)
            self.save_net(msg='', path=temp)

        for _ in range(iter):
            state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
            t_s = torch.tensor(state, dtype=torch.float).to(device)
            t_a_pos = self.torch_action2num(action).to(device)  # t_a是具体的物理动作，需要转换成动作编号作为索引值，是个tensor
            t_r = torch.unsqueeze(torch.tensor(reward, dtype=torch.float).to(device), dim=1)
            t_s_ = torch.tensor(new_state, dtype=torch.float).to(device)
            t_bool = torch.unsqueeze(torch.tensor(done, dtype=torch.float).to(device), dim=1)
            q_next = self.target_net(t_s_).detach().to(device)
            res = torch.max(input=q_next, dim=1, keepdim=True)
            q_target = t_r + self.gamma * (res[0].mul(t_bool))
            for _ in range(1):
                q_eval = self.eval_net(t_s).gather(1, t_a_pos.unsqueeze(1))
                loss = self.loss_func(q_eval, q_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_net(self, msg, path):
        torch.save(self.target_net.state_dict(), path + 'target' + msg)
        torch.save(self.eval_net.state_dict(), path + 'eval' + msg)

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
