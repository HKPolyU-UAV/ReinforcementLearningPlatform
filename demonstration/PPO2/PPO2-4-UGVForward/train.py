import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from UGVForward import UGVForward
from algorithm.policy_base.Proximal_Policy_Optimization2 import Proximal_Policy_Optimization2 as PPO2
from utils.functions import *
from utils.classes import Normalization

timestep = 0
ENV = 'UGVForward'
ALGORITHM = 'PPO2'
test_episode = []
test_reward = []
sumr_list = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)


# setup_seed(3407)


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 state_dim: int = 3,
                 action_dim: int = 3,
                 a_min: np.ndarray = np.zeros(3),
                 a_max: np.ndarray = np.ones(3),
                 init_std: float = 0.5,
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
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
        self.orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
        # mean = torch.relu(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action_mean = self.forward(t_state)
        return action_mean.detach().cpu().numpy().flatten()


class PPOCritic(nn.Module):
    def __init__(self, state_dim=3, use_orthogonal_init: bool = True):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
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


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)

    RETRAIN = False

    env = UGVForward()
    reward_norm = Normalization(shape=1)
    env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}
    t_epoch = 0  # 当前训练次数
    test_num = 0
    sumr = 0.
    buffer_index = 0
    ppo_msg = {'gamma': 0.99,
               'K_epochs': 25,
               'eps_clip': 0.2,
               'buffer_size': int(env.time_max / env.dt) * 4,
               'state_dim': env.state_dim,
               'action_dim': env.action_dim,
               'a_lr': 1e-4,
               'c_lr': 1e-3,
               'set_adam_eps': True,
               'lmd': 0.95,
               'use_adv_norm': True,
               'mini_batch_size': 64,
               'entropy_coef': 0.01,
               'use_grad_clip': False,
               'use_lr_decay': False,
               'max_train_steps': int(5e6),
               'using_mini_batch': False}

    std0 = (env.action_range[:, 1] - env.action_range[:, 0]) / 2 / 3
    agent = PPO2(env_msg=env_msg,
                 ppo_msg=ppo_msg,
                 actor=PPOActor_Gaussian(state_dim=env.state_dim,
                                         action_dim=env.action_dim,
                                         a_min=env.action_range[:, 0],
                                         a_max=env.action_range[:, 1],
                                         init_std=std0,
                                         use_orthogonal_init=True),
                 critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True))
    agent.PPO2_info()

    if RETRAIN:
        print('RELOADING......')
        '''如果两次奖励函数不一样，那么必须重新初始化 critic'''
        optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/trainNum_3650/'
        agent.actor.load_state_dict(torch.load(optPath + 'actor'))
        agent.critic.load_state_dict(torch.load(optPath + 'critic'))
        agent.critic.init(True)
        '''如果两次奖励函数不一样，那么必须重新初始化 critic'''

    env.is_terminal = True
    while True:
        '''1. 收集数据'''
        while buffer_index < agent.buffer.batch_size:
            if env.is_terminal:  # 如果某一个回合结束
                print('Sumr:  ', sumr)
                sumr_list.append(sumr)
                sumr = 0.
                env.reset(random=True)
            else:
                env.current_state = env.next_state.copy()
                a, a_log_prob = agent.choose_action(env.current_state)
                env.step_update(a)
                # env.visualization()
                sumr += env.reward

                if env.is_terminal:
                    if env.terminal_flag == 2:
                        success = 0
                    else:       # 只有回合结束，并且过早结束的时候，才是 1
                        success = 1
                else:
                    success = 0

                agent.buffer.append(s=env.current_state,
                                    a=a,
                                    log_prob=a_log_prob,
                                    r=reward_norm(env.reward),
                                    # r=env.reward,
                                    s_=env.next_state,
                                    done=1.0 if env.is_terminal else 0.0,   # 只要没有 s' 全都是 1
                                    success=success,                        #
                                    index=buffer_index)
                buffer_index += 1
        '''1. 收集数据'''

        '''2. 学习'''
        print('~~~~~~~~~~ Training Start~~~~~~~~~~')
        print('Train Epoch: {}'.format(t_epoch))
        timestep += ppo_msg['buffer_size']
        agent.learn(timestep, buf_num=1)
        agent.cnt += 1
        buffer_index = 0
        '''2. 学习'''

        '''3. 每学习 10 次，测试一下'''
        if t_epoch % 10 == 0 and t_epoch > 0:
            n = 5
            print('   Training pause......')
            print('   Testing...')
            for i in range(n):
                env.reset(random=True)
                test_r = 0.
                while not env.is_terminal:
                    env.current_state = env.next_state.copy()
                    _a = agent.evaluate(env.current_state)
                    env.step_update(_a)
                    test_r += env.reward
                    env.visualization()
                test_num += 1
                test_reward.append(test_r)
                print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
            pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
            pd.DataFrame({'sumr_list': sumr_list}).to_csv(simulationPath + 'sumr_list.csv')
            print('   Testing finished...')
            print('   Go back to training...')
        '''3. 每学习 10 次，测试一下'''

        '''4. 每学习 50 次，减小一次探索概率'''
        STD_DELAY_PER = 0.05
        if t_epoch % 200 == 0 and t_epoch > 0:
            _ratio = max(1 - t_epoch / 200 * STD_DELAY_PER, 0.05)
            agent.actor.std  = torch.tensor(std0 * _ratio, dtype=torch.float)
        '''4. 每学习 50 次，减小一次探索概率'''

        '''5. 每学习 50 次，保存一下 policy'''
        if t_epoch % 50 == 0 and t_epoch > 0:
            test_num += 1
            print('...check point save...')
            temp = simulationPath + 'trainNum_{}/'.format(t_epoch)
            os.mkdir(temp)
            time.sleep(0.01)
            agent.save_ac(msg='', path=temp)
        # env.save_state_norm(temp)
        '''5. 每学习 50 次，保存一下 policy'''

        t_epoch += 1
        print('~~~~~~~~~~  Training End ~~~~~~~~~~')
