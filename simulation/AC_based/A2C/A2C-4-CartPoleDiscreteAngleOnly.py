import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
from environment.envs.cartpole.cartpole_discrete_angle_only import CartPoleDiscreteAngleOnly
from algorithm.actor_critic.Advantage_AC import Advantage_AC as A2C
from common.common_func import *
from common.common_cls import *


# cfgPath = '../../../environment/config/'
# cfgFile = 'CartPoleDiscreteAngleOnly.xml'
optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
eval_rs = []
ALGORITHM = 'A2C'
ENV = 'CartPoleDiscreteAngleOnly'


class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_a2c'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_a2cALL'

        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        v = func.relu(self.fc1(state))  # relu
        v = func.relu(self.fc2(v))  # relu
        v = self.out(v)
        return v

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.out.reset_parameters()

    def save_checkpoint(self, name=None, path='', num=None):
        # print('...saving checkpoint...')
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
    def __init__(self, alpha, state_dim, action_num, name, chkpt_dir):
        super(SoftmaxActor, self).__init__()
        self.state_dim = state_dim
        self.action_num = action_num            # 这是一个list，每一个元素代表对应的action_space的长度，即 "每个action有几个取值"
        self.action_dim = len(action_num)       # 这是action的维度，即 "几个action"
        self.checkpoint_file = chkpt_dir + name + '_a2c'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_a2cALL'

        self.fc1 = nn.Linear(self.state_dim, 256)   # 公用层1
        self.fc2 = nn.Linear(256, 256)              # 公用层2
        self.out1 = nn.Linear(256, self.action_num[0])      # 离散倒立摆只有一个输入，所以只有out1
        '''
        如果有其他的action dim，那么就一个一个设计
        '''

        self.initialization_default()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'     # 推荐使用CPU
        self.to(self.device)

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.out1.reset_parameters()

    def forward(self, state):
        x = func.relu(self.fc1(state))
        x = func.relu(self.fc2(x))
        x = func.softmax(self.out1(x), dim=1)
        # TODO 这里可能出现维度问题
        return torch.cat((x, ), 1)

    def save_checkpoint(self, name=None, path='', num=None):
        # print('...saving checkpoint...')
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


if __name__ == '__main__':
    log_dir = '../../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN
    is_storage_only_success = False

    env = CartPoleDiscreteAngleOnly(initTheta=0, save_cfg=False)

    if TRAIN:
        actor = SoftmaxActor(1e-3, env.state_dim, env.action_num, 'SoftmaxActor', simulationPath)
        critic = Critic(1e-3, env.state_dim, env.action_dim, 'Critic', simulationPath)
        agent = A2C(env=env,
                    gamma=0.99,
                    timestep_num=10,
                    actor=actor,
                    critic=critic,
                    path=simulationPath)
        agent.A2C_info()

        successCounter = 0
        timeOutCounter = 0
        collisionCounter = 0
        # cv.waitKey(0)
        MAX_EPISODE = int(6e6)

        print('Start to train...')

        '''这并不是replay buffer，只是存储了10个timestep数据的小buffer'''
        '''需要注意的是：A2C训练别的env时，程序这里需要修改，因为这个环境的action是一维的，换别的不一定'''
        b_s = np.atleast_2d()
        b_a = np.atleast_2d()
        b_r = np.array([])
        b_lg_prob = np.atleast_2d()
        b_vs = np.array([])
        '''这并不是replay buffer，只是存储了10个timestep数据的小buffer'''
        while agent.episode <= MAX_EPISODE:
            # print('Episode:', agent.episode)
            R = 0
            step = 0
            env.reset_random()
            while not env.is_terminal:      # 每个回合
                env.current_state = env.next_state.copy()
                action_from_actor, lg_prob = agent.choose_action(env.current_state, False)   # 这是一个分布列
                done = 1 if env.is_terminal else 0

                action = agent.action_index_find(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action

                if step == 0:
                    b_s = np.atleast_2d(env.current_state.copy())
                    b_a = np.atleast_2d(action_from_actor.copy())
                    b_r = np.array(R + env.reward * agent.gamma * (1 - done))
                    b_lg_prob = np.atleast_2d(lg_prob)
                    b_vs = np.array(agent.critic(torch.FloatTensor(env.current_state).to(agent.device)).cpu().detach().numpy())
                else:
                    b_s = np.vstack((b_s, env.current_state.copy()))
                    b_a = np.vstack((b_a, action_from_actor.copy()))
                    b_r = np.vstack((b_r, R + env.reward * agent.gamma * (1 - done)))
                    b_lg_prob = np.vstack((b_lg_prob, lg_prob))
                    b_vs = np.vstack((b_vs, agent.critic(torch.FloatTensor(env.current_state).to(agent.device)).cpu().detach().numpy()))

                # env.show_dynamic_image(isWait=False)
                step += 1
                if step > 200:
                    break

            b_lg_prob = np.sum(b_lg_prob, 1)
            agent.learn(b_r, b_lg_prob, b_vs)  # TODO BUG

            agent.episode += 1
            if agent.episode % 50 == 0:
                temp = simulationPath + 'episode' + '_' + str(agent.episode) + '_save/'
                os.mkdir(temp)
                agent.actor.save_checkpoint(name='Actor_A2C', path=temp, num=agent.episode)
                agent.critic.save_checkpoint(name='Critic_A2C', path=temp, num=agent.episode)

            if agent.episode % 10 == 0:
                average_r, t = agent.agent_evaluate(5)
                eval_rs += average_r
                print('[%d], [%.3f], [%.3fs]' % (agent.episode, average_r, t))

    if TEST:
        pass
