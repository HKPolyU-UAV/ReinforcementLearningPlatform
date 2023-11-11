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


class ReplayBuffer:
    def __init__(self, max_size: int, batch_size: int, state_dim: int, action_dim: int):
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.end_mem = np.zeros(self.mem_size, dtype=np.float32)
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
        self.s = np.zeros((batch_size, state_dim))              # s
        self.a = np.zeros((batch_size, action_dim))             # a
        self.a_lp = np.zeros((batch_size, action_dim))                   # a_lp
        self.r = np.zeros((batch_size, 1))                           # r
        self.s_ = np.zeros((batch_size, state_dim))             # s'
        self.done = np.zeros((batch_size, 1))      # done
        self.success = np.zeros((batch_size, 1))      # success
        self.index = 0

    def append(self, s: np.ndarray, a: np.ndarray, log_prob: np.ndarray, r: float, s_: np.ndarray, done: float, success: float, index: int):
        self.s[index] = s
        self.a[index] = a
        self.a_lp[index] = log_prob
        self.r[index] = r
        self.s_[index] = s_
        self.done[index] = done
        self.success[index] = success

    def append_traj(self, s: np.ndarray, a: np.ndarray, log_prob: np.ndarray, r: np.ndarray, s_: np.ndarray, done: np.ndarray, success: np.ndarray):
        _l = len(done)
        for i in range(_l):
            if self.index == self.batch_size:
                self.index = 0
                return True
            else:
                self.s[self.index] = s[i]
                self.a[self.index] = a[i]
                self.a_lp[self.index] = log_prob[i]
                self.r[self.index] = r[i]
                self.s_[self.index] = s_[i]
                self.done[self.index] = done[i]
                self.success[self.index] = success[i]
                self.index += 1
        return False

    def to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_lp = torch.tensor(self.a_lp, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        success = torch.tensor(self.success, dtype=torch.float)

        return s, a, a_lp, r, s_, done, success

    def print_size(self):
        print('==== RolloutBuffer ====')
        print('actions: {}'.format(self.a.size))
        print('states: {}'.format(self.s.size))
        print('logprobs: {}'.format(self.a_lp.size))
        print('rewards: {}'.format(self.r.size))
        print('is_terminals: {}'.format(self.done.size))
        print('==== RolloutBuffer ====')


class RolloutBuffer2:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.s = np.atleast_2d([]).astype(np.float32)
        self.a = np.atleast_2d([]).astype(np.float32)
        self.a_lp = np.atleast_2d([]).astype(np.float32)
        self.r = np.atleast_2d([]).astype(np.float32)
        self.s_ = np.atleast_2d([]).astype(np.float32)
        self.done = np.atleast_2d([]).astype(np.float32)
        self.success = np.atleast_2d([]).astype(np.float32)
        self.index = 0

    def append_traj(self, s: np.ndarray, a: np.ndarray, log_prob: np.ndarray, r: np.ndarray, s_: np.ndarray, done: np.ndarray, success: np.ndarray):
        if self.index == 0:
            self.s = np.atleast_2d(s).astype(np.float32)
            self.a = np.atleast_2d(a).astype(np.float32)
            self.a_lp = np.atleast_2d(log_prob).astype(np.float32)
            self.r = np.atleast_2d(r).astype(np.float32)
            self.s_ = np.atleast_2d(s_).astype(np.float32)
            self.done = np.atleast_2d(done).astype(np.float32)
            self.success = np.atleast_2d(success).astype(np.float32)
        else:
            self.s = np.vstack((self.s, s))
            self.a = np.vstack((self.a, a))
            self.a_lp = np.vstack((self.a_lp, log_prob))
            self.r = np.vstack((self.r, r))
            self.s_ = np.vstack((self.s_, s_))
            self.done = np.vstack((self.done, done))
            self.success = np.vstack((self.success, success))
        self.index += len(done)

    def to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_lp = torch.tensor(self.a_lp, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        success = torch.tensor(self.success, dtype=torch.float)

        return s, a, a_lp, r, s_, done, success

    def print_size(self):
        print('==== RolloutBuffer ====')
        print('s: {}'.format(self.s.size))
        print('a: {}'.format(self.a.size))
        print('a_lp: {}'.format(self.a_lp.size))
        print('r: {}'.format(self.r.size))
        print('s_: {}'.format(self.s_.size))
        print('done: {}'.format(self.done.size))
        print('success: {}'.format(self.success.size))
        print('==== RolloutBuffer ====')

    def clean(self):
        self.index = 0
        self.s = np.atleast_2d([]).astype(np.float32)
        self.a = np.atleast_2d([]).astype(np.float32)
        self.a_lp = np.atleast_2d([]).astype(np.float32)
        self.r = np.atleast_2d([]).astype(np.float32)
        self.s_ = np.atleast_2d([]).astype(np.float32)
        self.done = np.atleast_2d([]).astype(np.float32)
        self.success = np.atleast_2d([]).astype(np.float32)


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
        # state_val = self.critic(s)

        return _a.detach(), action_log_prob.detach()  #, state_val.detach()

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


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 state_dim: int = 3,
                 action_dim: int = 3,
                 a_min: np.ndarray = np.zeros(3),
                 a_max: np.ndarray = np.ones(3),
                 init_std: float = 0.5,
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.mean_layer = nn.Linear(32, action_dim)
        # self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        # self.log_std = nn.Parameter(np.log(init_std) * torch.ones(action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = nn.Tanh()
        self.a_min = torch.tensor(a_min, dtype=torch.float)
        self.a_max = torch.tensor(a_max, dtype=torch.float)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.action_dim = action_dim
        self.std = torch.tensor(init_std, dtype=torch.float)

        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.fc3)
        self.orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        # mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
        mean = torch.relu(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        # mean = torch.tensor(mean, dtype=torch.float)
        # log_std = self.log_std.expand_as(mean)
        # std = torch.exp(log_std)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)  # Get the Gaussian distribution
        # std = self.std.expand_as(mean)
        # dist = Normal(mean, std)
        return dist

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action_mean = self.forward(t_state)
        return action_mean.detach().cpu().numpy().flatten()


class PPOCritic(nn.Module):
    def __init__(self, state_dim=3, use_orthogonal_init: bool = True):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activate_func = nn.Tanh()

        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s

    def init(self, use_orthogonal_init):
        if use_orthogonal_init:
            self.orthogonal_init_all()
        else:
            self.fc1.reset_parameters()
            self.fc2.reset_parameters()
            self.fc3.reset_parameters()


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating, update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


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
