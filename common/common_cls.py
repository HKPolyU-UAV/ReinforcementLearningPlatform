# import math
import random
# import re
import numpy as np
# from numpy import linalg
import torch.nn as nn
import torch.nn.functional as func
import torch
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


# import scipy.spatial as spt


class ReplayBuffer:
    def __init__(self, max_size: int, batch_size: int, state_dim: int, action_dim: int):
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.end_mem = np.zeros(self.mem_size, dtype=np.float)
        self.log_prob_mem = np.zeros(self.mem_size)
        self.sorted_index = []
        self.resort_count = 0

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, state_: np.ndarray, done: float, log_p: float = 0., has_log_prob: bool = False):
        index = self.mem_counter % self.mem_size
        self.s_mem[index] = state
        self.a_mem[index] = action
        self.r_mem[index] = reward
        self._s_mem[index] = state_
        self.end_mem[index] = 1 - done
        if has_log_prob:
            self.log_prob_mem[index] = log_p
        self.mem_counter += 1

    def get_reward_sort(self):
        """
        :return:        根据奖励大小得到所有数据的索引值，升序，即从小到大
        """
        print('...sorting...')
        self.sorted_index = sorted(range(min(self.mem_counter, self.mem_size)), key=lambda k: self.r_mem[k], reverse=False)

    def store_transition_per_episode(self, states, actions, rewards, states_, dones, log_ps=None, has_log_prob: bool = False):
        self.resort_count += 1
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i], dones[i], log_ps, has_log_prob)

    def sample_buffer(self, is_reward_ascent: bool = True, has_log_prob: bool = False):
        max_mem = min(self.mem_counter, self.mem_size)
        if is_reward_ascent:
            batchNum = min(int(0.25 * max_mem), self.batch_size)
            batch = random.sample(self.sorted_index[-int(0.25 * max_mem):], batchNum)
        else:
            batch = np.random.choice(max_mem, self.batch_size)
        states = self.s_mem[batch]
        actions = self.a_mem[batch]
        rewards = self.r_mem[batch]
        actions_ = self._s_mem[batch]
        terminals = self.end_mem[batch]
        if has_log_prob:
            log_probs = self.log_prob_mem[batch]
            return states, actions, rewards, actions_, terminals, log_probs
        else:
            return states, actions, rewards, actions_, terminals


class RolloutBuffer:
    def __init__(self, batch_size: int, state_dim: int, action_dim: int):
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = np.zeros((batch_size, action_dim))
        self.states = np.zeros((batch_size, state_dim))
        self.log_probs = np.zeros(batch_size)
        self.rewards = np.zeros(batch_size)
        self.state_values = np.zeros(batch_size)
        self.is_terminals = np.zeros(batch_size, dtype=np.float)
        self.index = 0

    def append(self, s: np.ndarray, a: np.ndarray, log_prob: float, r: float, sv: float, done: float, index: int):
        self.actions[index] = a
        self.states[index] = s
        self.log_probs[index] = log_prob
        self.rewards[index] = r
        self.state_values[index] = sv
        self.is_terminals[index] = done

    def append_traj(self, s: np.ndarray, a: np.ndarray, log_prob: np.ndarray, r: np.ndarray, sv: np.ndarray, done: np.ndarray):
        _l = len(done)
        for i in range(_l):
            if self.index == self.batch_size:
                self.index = 0
                return True
            else:
                self.actions[self.index] = a[i]
                self.states[self.index] = s[i]
                self.log_probs[self.index] = log_prob[i]
                self.rewards[self.index] = r[i]
                self.state_values[self.index] = sv[i]
                self.is_terminals[self.index] = done[i]
                self.index += 1
        return False

    def print_size(self):
        print('==== RolloutBuffer ====')
        print('actions: {}'.format(self.actions.size))
        print('states: {}'.format(self.states.size))
        print('logprobs: {}'.format(self.log_probs.size))
        print('rewards: {}'.format(self.rewards.size))
        print('state_values: {}'.format(self.state_values.size))
        print('is_terminals: {}'.format(self.is_terminals.size))
        print('==== RolloutBuffer ====')


class RolloutBuffer2:
    def __init__(self, state_dim: int, action_dim: int):
        self.buffer_size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = np.atleast_2d([]).astype(np.float32)
        self.states = np.atleast_2d([]).astype(np.float32)
        self.log_probs = np.atleast_1d([]).astype(np.float32)
        self.rewards = np.atleast_1d([]).astype(np.float32)
        self.state_values = np.atleast_1d([]).astype(np.float32)
        self.is_terminals = np.atleast_1d([]).astype(np.float32)

    def append_traj(self, s: np.ndarray, a: np.ndarray, log_prob: np.ndarray, r: np.ndarray, sv: np.ndarray, done: np.ndarray):
        if self.buffer_size == 0:
            self.states = np.atleast_2d(s).astype(np.float32)
            self.actions = np.atleast_2d(a).astype(np.float32)
            self.log_probs = np.atleast_1d(log_prob).astype(np.float32)
            self.rewards = np.atleast_1d(r).astype(np.float32)
            self.state_values = np.atleast_1d(sv).astype(np.float32)
            self.is_terminals = np.atleast_1d(done).astype(np.float32)
        else:
            self.states = np.vstack((self.states, s))
            self.actions = np.vstack((self.actions, a))
            self.log_probs = np.hstack((self.log_probs, log_prob))
            self.rewards = np.hstack((self.rewards, r))
            self.state_values = np.hstack((self.state_values, sv))
            self.is_terminals= np.hstack((self.is_terminals, done))
        self.buffer_size += len(done)

    def print_size(self):
        print('==== RolloutBuffer ====')
        print('actions: {}'.format(self.actions.size))
        print('states: {}'.format(self.states.size))
        print('logprobs: {}'.format(self.log_probs.size))
        print('rewards: {}'.format(self.rewards.size))
        print('state_values: {}'.format(self.state_values.size))
        print('is_terminals: {}'.format(self.is_terminals.size))
        print('==== RolloutBuffer ====')

    def clean(self):
        self.buffer_size = 0
        self.actions = np.atleast_2d([]).astype(np.float32)
        self.states = np.atleast_2d([]).astype(np.float32)
        self.log_probs = np.atleast_1d([]).astype(np.float32)
        self.rewards = np.atleast_1d([]).astype(np.float32)
        self.state_values = np.atleast_1d([]).astype(np.float32)
        self.is_terminals = np.atleast_1d([]).astype(np.float32)


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class GaussianNoise(object):
    def __init__(self, mu):
        self.mu = mu
        self.sigma = 1 / 3

    def __call__(self, sigma=1 / 3):
        return np.random.normal(self.mu, sigma, self.mu.shape)


class DQNNet(nn.Module):
    def __init__(self, _input: int = 1, _output: list = None):
        """
        :brief:             神经网络初始化
        :param _input:      输入维度
        :param _output:     输出维度
        """
        super(DQNNet, self).__init__()
        self.hidden1 = nn.Linear(_input, 64)  # input -> hidden1
        self.hidden2 = nn.Linear(64, 64)  # hidden1 -> hidden2
        self.out = nn.Linear(64, _output[0])  # hidden2 -> output
        self.init()

    def init(self):
        torch.nn.init.orthogonal_(self.hidden1.weight, gain=1)
        torch.nn.init.uniform_(self.hidden1.bias, 0, 1)
        torch.nn.init.orthogonal_(self.hidden2.weight, gain=1)
        torch.nn.init.uniform_(self.hidden2.bias, 0, 1)
        torch.nn.init.orthogonal_(self.out.weight, gain=1)
        torch.nn.init.uniform_(self.out.bias, 0, 1)

    def forward(self, _x):
        """
        :brief:         神经网络前向传播
        :param _x:      输入网络层的张量
        :return:        网络的输出
        """
        x = _x
        x = self.hidden1(x)
        x = func.relu(x)
        x = self.hidden2(x)
        x = func.relu(x)
        state_action_value = self.out(x)
        return state_action_value


class DuelingNeuralNetwork(nn.Module):
    def __init__(self, _input: int, _output: list):
        """
        :brief:             神经网络初始化
        :param _input:      输入维度
        :param _output:     输出维度
        """
        super(DuelingNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(_input, 64)  # input -> hidden1
        self.hidden2 = nn.Linear(64, 64)  # hidden1 -> hidden2
        # self.out = nn.Linear(64, _output)  # hidden2 -> output
        self.value = nn.Linear(64, _output[0])
        self.advantage = nn.Linear(64, _output[0])
        # self.init()
        self.init_default()

    def init(self):
        torch.nn.init.orthogonal_(self.hidden1.weight, gain=1)
        torch.nn.init.uniform_(self.hidden1.bias, 0, 1)
        torch.nn.init.orthogonal_(self.hidden2.weight, gain=1)
        torch.nn.init.uniform_(self.hidden2.bias, 0, 1)
        torch.nn.init.orthogonal_(self.out.weight, gain=1)
        torch.nn.init.uniform_(self.out.bias, 0, 1)
        torch.nn.init.orthogonal_(self.value.weight, gain=1)
        torch.nn.init.uniform_(self.value.bias, 0, 1)
        torch.nn.init.orthogonal_(self.advantage.weight, gain=1)
        torch.nn.init.uniform_(self.advantage.bias, 0, 1)

    def init_default(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.value.reset_parameters()
        self.advantage.reset_parameters()

    def forward(self, _x):
        """
        :brief:         神经网络前向传播
        :param _x:      输入网络层的张量
        :return:        网络的输出
        """
        x = _x
        x = self.hidden1(x)
        x = func.relu(x)
        x = self.hidden2(x)
        x = func.relu(x)

        x1 = self.value(x)
        x1 = func.relu(x1)

        x2 = self.advantage(x)
        x2 = func.relu(x2)

        state_action_value = x1 + (x2 - x2.mean())
        return state_action_value


class Critic(nn.Module):
    def __init__(self, beta=1e-3, state_dim=1, action_dim=1, name='Critic', chkpt_dir=''):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'
        self.layer = nn.Linear(state_dim + action_dim, 1)  # layer for the first Q
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action_value = self.layer(torch.cat([state, action], 1))

        return state_action_value

    def initialization(self):
        pass

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DualCritic(nn.Module):
    def __init__(self, beta=1e-3, state_dim=1, action_dim=1, name='DualCritic', chkpt_dir=''):
        super(DualCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.l11 = nn.Linear(state_dim + action_dim, 1)  # layer for the first Q
        self.l21 = nn.Linear(state_dim + action_dim, 1)  # layer for the second Q
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)  # optimizer for Q

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action_value1 = self.l11(torch.cat([state, action], 1))
        state_action_value2 = self.l21(torch.cat([state, action], 1))

        return state_action_value1, state_action_value2

    def initialization(self):
        pass

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Actor(nn.Module):
    def __init__(self, alpha=1e-4, state_dim=1, action_dim=1, name='Actor', chkpt_dir=''):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'
        self.layer = nn.Linear(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        pass

    def forward(self, state):
        # x = torch.tanh(state)  # bound the output to [-1, 1]
        # return x
        pass

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ProbActor(nn.Module):
    def __init__(self, alpha=1e-4, state_dim=1, action_dim=1, name='ProbActor', chkpt_dir=''):
        super(ProbActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.l1 = nn.Linear(state_dim, 64)  # The layer for mean output
        self.mean_layer = nn.Linear(64, action_dim)  # The layer for mean output
        self.log_std_layer = nn.Linear(64, action_dim)  # THe layer for log-std output

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        pass

    def forward(self, state, deterministic=False):
        x = func.relu(self.l1(state))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 20)
        std = torch.exp(log_std)

        gaussian_dist = Normal(mean, std)

        if deterministic:
            act = mean
        else:
            act = gaussian_dist.rsample()

        log_pi = gaussian_dist.log_prob(act).sum(dim=1, keepdim=True)
        log_pi -= (2 * (np.log(2) - act - func.softplus(-2 * act))).sum(dim=1, keepdim=True)

        act = torch.tanh(act)  # 策略的输出必须被限制在 [-1, 1] 之间

        return act, log_pi

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class SoftmaxActor(nn.Module):
    def __init__(self, alpha=1e-4, state_dim=1, action_dim=1, action_num=None, name='DiscreteActor', chkpt_dir=''):
        super(SoftmaxActor, self).__init__()
        self.state_dim = state_dim              # 状态的维度，即 ”有几个状态“
        self.action_dim = action_dim            # 动作的维度，即 "有几个动作"
        if action_num is None:
            self.action_num = [3, 3, 3, 3]      # 每个动作有几个取值，离散动作空间特有
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_PPO_Dis'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_PPO_DisALL'

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = [nn.Linear(64, self.action_num[i]) for i in range(self.action_dim)]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

        self.initialization()

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def initialization(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        for i in range(self.action_dim):
            self.orthogonal_init(self.out[i], gain=0.01)

    def forward(self, xx: torch.Tensor):
        xx = torch.tanh(self.fc1(xx))       # xx -> 第一层 -> tanh
        xx = torch.tanh(self.fc2(xx))       # xx -> 第二层 -> tanh
        a_prob = []
        for i in range(self.action_dim):
            a_prob.append(func.softmax(self.out[i](xx), dim=1).T)   # xx -> 每个动作维度的第三层 -> softmax
        return nn.utils.rnn.pad_sequence(a_prob).T      # 得到很多分布列，分布列合并，差的数用 0 补齐，不影响 log_prob 和 entropy

    def evaluate(self, xx: torch.Tensor):
        xx = torch.unsqueeze(xx, 0)
        a_prob = self.forward(xx)
        _a = torch.argmax(a_prob, dim=2)
        return _a

    def choose_action(self, xx):  # choose action 默认是在训练情况下的函数，默认有batch
        # xx = torch.unsqueeze(xx, 0)
        with torch.no_grad():
            dist = Categorical(probs=self.forward(xx))
            _a = dist.sample()
            _a_logprob = dist.log_prob(_a)
            _a_entropy = dist.entropy()
        '''
            这里跟连续系统不一样的地方在于，这里的概率是多个分布列，pytorch 或许无法表示多维分布列。
            所以用了 mean 函数，但是主观分析不影响结果，因为 mean 的单调性与 sum 是一样的。
            连续动作有多维联合高斯分布，但是协方差矩阵都是对角阵，所以跟多个一维的也没区别。
        '''
        return _a, torch.mean(_a_logprob, dim=1), torch.mean(_a_entropy, dim=1)
        # return _a

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


# class SoftmaxActor(nn.Module):
#     def __init__(self, alpha=3e-4, state_dim=1, action_dim=1, action_num=None, name='DiscreteActor', chkpt_dir=''):
#         super(SoftmaxActor, self).__init__()
#         self.state_dim = state_dim              # 状态的维度，即 ”有几个状态“
#         self.action_dim = action_dim            # 动作的维度，即 "有几个动作"
#         if action_num is None:
#             self.action_num = [3, 3, 3, 3]      # 每个动作有几个取值，离散动作空间特有
#         self.alpha = alpha
#         self.checkpoint_file = chkpt_dir + name + '_PPO_Dis'
#         self.checkpoint_file_whole_net = chkpt_dir + name + '_PPO_DisALL'
#
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.out = [nn.Linear(64, 2) for _ in range(self.action_dim)]
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
#
#         self.initialization()
#
#         # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.device = 'cpu'
#         self.to(self.device)
#
#     @staticmethod
#     def orthogonal_init(layer, gain=1.0):
#         nn.init.orthogonal_(layer.weight, gain=gain)
#         nn.init.constant_(layer.bias, 0)
#
#     def initialization(self):
#         self.orthogonal_init(self.fc1)
#         self.orthogonal_init(self.fc2)
#         for i in range(self.action_dim):
#             self.orthogonal_init(self.out[i], gain=0.01)
#
#     def forward(self, xx: torch.Tensor):
#         xx = torch.tanh(self.fc1(xx))       # xx -> 第一层 -> tanh
#         xx = torch.tanh(self.fc2(xx))       # xx -> 第二层 -> tanh
#         a_prob = []
#         for i in range(self.action_dim):
#             a_prob.append(func.softmax(self.out[i](xx), dim=1).T)   # xx -> 每个动作维度的第三层 -> softmax
#         return nn.utils.rnn.pad_sequence(a_prob).T      # 得到很多分布列，分布列合并，差的数用 0 补齐，不影响 log_prob 和 entropy
#
#     def evaluate(self, xx: torch.Tensor):
#         xx = torch.unsqueeze(xx, 0)
#         a_prob = self.forward(xx)
#         _a = torch.argmax(a_prob, dim=2)
#         return _a
#
#     def choose_action(self, xx):  # choose action 默认是在训练情况下的函数，默认有batch
#         xx = torch.unsqueeze(xx, 0)
#         with torch.no_grad():
#             dist = Categorical(probs=self.forward(xx))
#             _a = dist.sample()
#             _a_logprob = dist.log_prob(_a)
#             _a_entropy = dist.entropy()
#         '''
#             这里跟连续系统不一样的地方在于，这里的概率是多个分布列，pytorch 或许无法表示多维分布列。
#             所以用了 sum 函数，但是主观分析不影响结果，因为 sum 的单调性与 sum 是一样的。
#             连续动作有多维联合高斯分布，但是协方差矩阵都是对角阵，所以跟多个一维的也没区别。
#         '''
#         return _a, torch.sum(_a_logprob, dim=1), torch.sum(_a_entropy, dim=1)
#         # return _a
#
#     def save_checkpoint(self, name=None, path='', num=None):
#         print('...saving checkpoint...')
#         if name is None:
#             torch.save(self.state_dict(), self.checkpoint_file)
#         else:
#             if num is None:
#                 torch.save(self.state_dict(), path + name)
#             else:
#                 torch.save(self.state_dict(), path + name + str(num))
#
#     def save_all_net(self):
#         print('...saving all net...')
#         torch.save(self, self.checkpoint_file_whole_net)
#
#     def load_checkpoint(self):
#         print('...loading checkpoint...')
#         self.load_state_dict(torch.load(self.checkpoint_file))


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
        super(PPOActorCritic, self).__init__()
        self.checkpoint_file = chkpt_dir + name + '_ppo'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
        self.action_dim = _action_dim
        # 初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
        self.action_var = torch.full((_action_dim,), _action_std_init * _action_std_init)
        self.actor = nn.Sequential(
            nn.Linear(_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, _action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.device = 'cpu'
        # torch.cuda.empty_cache()
        self.to(self.device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, s: torch.Tensor) -> tuple:
        action_mean = self.actor(s)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)     # 多变量高斯分布，均值，方差

        _a = dist.sample()
        action_log_prob = dist.log_prob(_a)
        state_val = self.critic(s)

        return _a.detach(), action_log_prob.detach(), state_val.detach()

    def evaluate(self, s, a):
        action_mean = self.actor(s)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            a = a.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(a)
        dist_entropy = dist.entropy()
        state_values = self.critic(s)

        return action_logprobs, state_values, dist_entropy

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)                   # 原来就是0 照着另一个程序改的
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['step'].share_memory_()       # 这句话是对照另一个程序加的
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
