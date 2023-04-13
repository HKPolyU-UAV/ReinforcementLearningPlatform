import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np
import torch
import visdom

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
from environment.envs.UGV2.UGVBidirectional import UGV_Bidirectional as env
from algorithm.policy_base.Proximal_Policy_Optimization_Discrete import Proximal_Policy_Optimization_Discrete as PPO_Dis
from common.common_cls import *

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ALGORITHM = 'PPO_Discrete'
ENV = 'UGVBidirectional'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(3407)


class SoftmaxActor(nn.Module):
    def __init__(self, alpha=3e-4, state_dim=1, action_dim=1, action_num=None, name='DiscreteActor', chkpt_dir=''):
        super(SoftmaxActor, self).__init__()
        self.state_dim = state_dim              # 状态的维度，即 ”有几个状态“
        self.action_dim = action_dim            # 动作的维度，即 "有几个动作"
        if action_num is None:
            self.action_num = [2, 2, 2, 2]      # 每个动作有几个取值，离散动作空间特有
        self.index = [0]
        for i in range(action_dim):
            self.index.append(self.index[i] + env.action_num[i])
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_PPO_Dis'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_PPO_DisALL'

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, sum(env.action_num))
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
        self.orthogonal_init(self.out)

    def forward(self, xx: torch.Tensor):
        xx = torch.tanh(self.fc1(xx))       # xx -> 第一层 -> tanh
        xx = torch.tanh(self.fc2(xx))       # xx -> 第二层 -> tanh
        a_prob = []
        for i in range(self.action_dim):
            a_prob.append(func.softmax(xx[:, self.index[i]:self.index[i + 1]], dim=1).T)
        return nn.utils.rnn.pad_sequence(a_prob).T      # 得到很多分布列，分布列合并，差的数用 0 补齐，不影响 log_prob 和 entropy

    def evaluate(self, xx: torch.Tensor):
        xx = torch.unsqueeze(xx, 0)
        a_prob = self.forward(xx)
        _a = torch.argmax(a_prob, dim=2)
        return _a

    def choose_action(self, xx):  # choose action 默认是在训练情况下的函数，默认有batch
        xx = torch.unsqueeze(xx, 0)
        with torch.no_grad():
            dist = Categorical(probs=self.forward(xx))
            _a = dist.sample()
            _a_logprob = dist.log_prob(_a)
            _a_entropy = dist.entropy()
        '''
            这里跟连续系统不一样的地方在于，这里的概率是多个分布列，pytorch 或许无法表示多维分布列。
            所以用了 sum 函数，但是主观分析不影响结果，因为 sum 的单调性与 mean 是一样的。
            连续动作有多维联合高斯分布，但是协方差矩阵都是对角阵，所以跟多个一维的也没区别。
        '''
        return _a, torch.sum(_a_logprob, dim=1), torch.sum(_a_entropy, dim=1)

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


class Critic(nn.Module):
    def __init__(self, beta=1e-3, state_dim=1, action_dim=1, name='Critic', chkpt_dir=''):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.checkpoint_file = chkpt_dir + name + '_PPO_Critic'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_PPO_CriticALL'

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, xx):
        xx = torch.tanh(self.fc1(xx))  # xx -> 第一层 -> tanh
        xx = torch.tanh(self.fc2(xx))  # xx -> 第二层 -> tanh
        xx = self.fc3(xx)
        return xx

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def initialization(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.fc3)

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


if __name__ == '__main__':
    log_dir = '../../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)

    env = env(pos0=np.array([0.5, 0.5]),
              vel0=0.,
			  phi0=0.,
			  omega0=0.,
              map_size=np.array([5.0, 5.0]),
              target=np.array([4.5, 4.5]),
              is_controller_BangBang=True)

    vis = visdom.Visdom(env=ALGORITHM + '4' + ENV)

    '''重新加载Policy网络结构，这是必须的操作'''
    # 3e-4 for actor  first time
    # 3e-4 for critic first time
    actor = SoftmaxActor(1e-4, env.state_dim, env.action_dim, env.action_num, 'DiscreteActor', simulationPath)
    critic = Critic(3e-3, env.state_dim, env.action_dim, 'Critic', simulationPath)
    agent = PPO_Dis(env=env,
                    gamma=0.99,
                    K_epochs=10,
                    eps_clip=0.2,
                    buffer_size=int(env.timeMax / env.dt * 4),  # 假设可以包含两条完整的最长时间的轨迹
                    actor=actor,
                    critic=critic,
                    path=simulationPath)
    '''重新加载Policy网络结构，这是必须的操作'''
    agent.PPO_info()

    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    if TRAIN:
        if RETRAIN:
            agent.actor.load_state_dict(torch.load('Actor_PPO5750'))
            agent.critic.load_state_dict(torch.load('Critic_PPO5750'))

        max_training_timestep = int(env.timeMax / env.dt) * 20000  # 10000回合

        train_num = 0
        index = 0
        evaluate_r = np.array([])
        evaluate_e = np.array([])
        complete_traj_count = 0
        # while timestep <= max_training_timestep:
        while train_num < 10000:
            '''存数'''
            while index < agent.buffer.batch_size:
                if index == agent.buffer.batch_size:
                    break
                env.reset_random()
                # env.reset()
                sumr = 0
                while not env.is_terminal:
                    if index == agent.buffer.batch_size:
                        break
                    env.current_state = env.next_state.copy()
                    action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state, exploration=-1)  # 返回三个没有梯度的tensor
                    action_from_actor = action_from_actor.numpy()
                    action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
                    env.step_update(action)  # 环境更新的action需要是物理的action
                    sumr += env.reward
                    agent.buffer.append(s=env.current_state,
                                        a=action_from_actor,
                                        log_prob=a_log_prob.numpy(),
                                        r=env.reward,
                                        sv=s_value.numpy(),
                                        done=1.0 if env.is_terminal else 0.0,
                                        index=index)
                    index += 1
                    timestep += 1
                '''One trajectory, complete'''
                if env.time >= env.timeMax:     # which means the trajectory is complete
                    average_r = sumr / env.time
                    vis.line(X=np.array([complete_traj_count]), Y=np.array([average_r]),
                             win='trajectory average reward',
                             update='append' if complete_traj_count > 0 else None,
                             opts=dict(title='trajectory average reward'))
                    complete_traj_count += 1
                '''One trajectory, complete'''
            '''存数'''
            '''学习'''
            print('========== LEARN ==========')
            print('Episode: {}'.format(agent.episode))
            print('Num of learning: {}'.format(train_num))
            agent.learn(adv_norm=False)     # default: False
            train_num += 1
            index = 0
            if train_num % 50 == 0 and train_num > 0:    # '1' should be 50
                # agent.agent_evaluate_once()
                rr, ee = agent.agent_evaluate(False)
                print('----- position errors -----')
                print('Training num:  ', train_num)
                print(ee)
                print('----- position errors -----')
                if train_num == 50:
                    evaluate_r = rr.copy()
                    evaluate_e = ee.copy()
                else:
                    evaluate_r = np.hstack((evaluate_r, rr))
                    evaluate_e = np.hstack((evaluate_e, ee))
                xx = np.arange(int(train_num / 50 - 1) * len(ee), int(train_num / 50) * len(ee), 1)
                # vis.line(X=xx, Y=rr, win='reward',
                #          update='append' if train_num > 50 else None, opts=dict(title='reward'))
                vis.line(X=xx, Y=ee, win='position error',
                         update='append' if train_num > 50 else None, opts=dict(title='position error'))
                dir_temp = simulationPath + 'training' + '_' + str(train_num) + '_save/'
                os.mkdir(dir_temp)
                time.sleep(0.01)
                agent.actor.save_checkpoint(name='Actor_PPO', path=dir_temp, num=train_num)
                agent.critic.save_checkpoint(name='Critic_PPO', path=dir_temp, num=train_num)
            print('========== LEARN ==========')
            '''学习'''
    if TEST:
        # agent.actor.load_state_dict(torch.load(optPath + ALGORITHM + '-' + ENV + '/' + 'Actor_PPO'))
        agent.actor.load_state_dict(torch.load('Actor_PPO5750'))
        rr, ee = agent.agent_evaluate(True, _random=True, test_num=50)
        print(rr)
        print(ee)
