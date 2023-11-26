import os
import sys
import datetime
import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from UGVBidirectional import UGVBidirectional as env
from algorithm.actor_critic.Soft_Actor_Critic import SAC
from utils.functions import *
from utils.classes import Normalization

timestep = 0
ENV = 'UGVBidirectional'
ALGORITHM = 'SAC'
MAX_EPISODE = 1500
r_norm = Normalization(shape=1)
"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class SACActor(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, a_min: np.ndarray = np.zeros(3),
                 a_max: np.ndarray = np.ones(3), std_min: float = 0.05, std_scale: float = 2.,
                 use_orthogonal_init: bool = True):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        self.device = device
        self.a_min = torch.tensor(a_min, dtype=torch.float).to(self.device)
        self.a_max = torch.tensor(a_max, dtype=torch.float).to(self.device)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.std_min = std_min
        self.std_scale = std_scale
        self.to(device)
        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.mean_layer, gain=0.01)
        self.orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std,
                              torch.log(self.std_min * (self.a_max - self.a_min) / 2),
                              (self.a_max - self.a_min) / 2 / self.std_scale)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - func.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = torch.tanh(a) * self.gain + self.off

        return a, log_pi


class SACCritic(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 1, use_orthogonal_init: bool = True):
        super(SACCritic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        self.device = device
        self.to(self.device)
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
        self.orthogonal_init(self.fc4)
        self.orthogonal_init(self.fc5)
        self.orthogonal_init(self.fc6)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = torch.relu(self.fc1(s_a))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.relu(self.fc4(s_a))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


def fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_dw = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset(randomEnv)
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_dw.clear()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action = agent.choose_action(env.current_state, deterministic=True)
            env.step_update(_action)
            # env.visualization()
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_dw.append(0.0 if env.is_terminal and env.terminal_flag != 3 else 1.0)
            else:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state,
                                              0.0 if env.is_terminal and env.terminal_flag != 3 else 1.0)
        if is_only_success:
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_dw)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        env.reset(randomEnv)
        while not env.is_terminal:
            if agent.memory.mem_counter % 1000 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action = agent.choose_action_random()
            env.step_update(_action)
            # env.visualization()
            # if env.reward > 0:
            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state,
                                          0.0 if env.is_terminal and env.terminal_flag != 3 else 1.0)


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                          '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)

    RETRAIN = True

    env = env()
    actor = SACActor(env.state_dim, env.action_dim, env.action_range[:, 0], env.action_range[:, 1], std_scale=1.)
    critic = SACCritic(env.state_dim, env.action_dim)
    target_critic = SACCritic(env.state_dim, env.action_dim)
    actor_lr, critic_lr, alpha_lr = 1e-4, 1e-4, 1e-4
    env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'action_range': env.action_range, 'name': ENV}
    agent = SAC(env_msg=env_msg,
                gamma=0.99,
                critic_tau=0.005,
                memory_capacity=1000000,
                batch_size=256,
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                a_lr=actor_lr,
                c_lr=critic_lr,
                alpha_lr=alpha_lr,
                adaptive_alpha=True)
    agent.SAC_info()

    if RETRAIN:
        print('RELOADING......')
        optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
        agent.actor.load_state_dict(torch.load('actor'))
        agent.critic.load_state_dict(torch.load('critic'))
        agent.target_critic.load_state_dict(torch.load('target_critic'))
        agent.critic.orthogonal_init_all()
        agent.target_critic.orthogonal_init_all()
        fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.025, is_only_success=False)
    else:
        '''fullFillReplayMemory_Random'''
        fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.025)
        '''fullFillReplayMemory_Random'''

    print('Start to train...')
    new_state, new_action, new_reward, new_state_, new_dw = [], [], [], [], []
    step = 0
    is_storage_only_success = False
    while agent.episode <= MAX_EPISODE:
        env.reset(True)
        sumr = 0
        new_state.clear()
        new_action.clear()
        new_reward.clear()
        new_state_.clear()
        new_dw.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            if np.random.uniform(0, 1) < 0.00:
                action = agent.choose_action_random()  # 有一定探索概率完全随机探索
            else:
                action = agent.choose_action(env.current_state)
            env.step_update(action)
            step += 1
            if agent.episode % 10 == 0:
                env.visualization()
            sumr = sumr + env.reward
            if is_storage_only_success:
                new_state.append(env.current_state)
                new_action.append(env.current_action)
                new_reward.append(env.reward)
                new_state_.append(env.next_state)
                new_dw.append(0.0 if env.is_terminal and env.terminal_flag != 3 else 1.0)
            else:
                # if env.reward > 0:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state,
                                              0.0 if env.is_terminal and env.terminal_flag != 3 else 1.0)
            agent.learn(is_reward_ascent=False)
        '''跳出循环代表回合结束'''
        if is_storage_only_success:
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_dw)
        '''跳出循环代表回合结束'''
        print('Episode:', agent.episode, 'Cumulative reward:', round(sumr, 3))
        agent.episode += 1
        if agent.episode % 30 == 0:
            temp = simulationPath + 'trainNum_{}/'.format(agent.episode)
            os.mkdir(temp)
            time.sleep(0.01)
            print('Save net', agent.episode)
            agent.save_ac(msg='', path=temp)
