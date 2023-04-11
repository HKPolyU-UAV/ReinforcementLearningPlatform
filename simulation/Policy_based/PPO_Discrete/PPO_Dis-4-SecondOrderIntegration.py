import os
import sys
import datetime
import time
import cv2 as cv
import visdom

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
from environment.envs.SecondOrderIntegration.SecondOrderIntegration import SecondOrderIntegration as env
from algorithm.policy_base.Proximal_Policy_Optimization_Discrete import Proximal_Policy_Optimization_Discrete as PPO_Dis
from common.common_cls import *

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ALGORITHM = 'PPO_Discrete'
ENV = 'SecondOrderIntegration'


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
            self.action_num = [3, 3, 3, 3]      # 每个动作有几个取值，离散动作空间特有
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_PPO_Dis'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_PPO_DisALL'

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = [nn.Linear(64, env.action_num[i]) for i in range(self.action_dim)]
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
        xx = torch.unsqueeze(xx, 0)
        with torch.no_grad():
            dist = Categorical(probs=self.forward(xx))
            _a = dist.sample()
            _a_logprob = dist.log_prob(_a)
            _a_entropy = dist.entropy()
        '''
            这里跟连续系统不一样的地方在于，这里的概率是多个分布列，pytorch 或许无法表示多维分布列。
            所以用了 sum 函数，但是主观分析不影响结果，因为 sum 的单调性与 sum 是一样的。
            连续动作有多维联合高斯分布，但是协方差矩阵都是对角阵，所以跟多个一维的也没区别。
        '''
        return _a, torch.sum(_a_logprob, dim=1), torch.sum(_a_entropy, dim=1)
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
    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    env = env(pos0=np.array([2.5, 2.5]),
              vel0=np.array([0.0, 0.0]),
              map_size=np.array([5.0, 5.0]),
              target=np.array([4.5, 4.5]),
              is_controller_BangBang=True)

    vis = visdom.Visdom(env=ALGORITHM + '4' + ENV)

    if TRAIN:
        '''重新加载Policy网络结构，这是必须的操作'''
        actor = SoftmaxActor(3e-4, env.state_dim, env.action_dim, env.action_num, 'DiscreteActor', simulationPath)
        critic = Critic(3e-4, env.state_dim, env.action_dim, 'Critic', simulationPath)
        agent = PPO_Dis(env=env,
                        gamma=0.99,
                        K_epochs=40,
                        eps_clip=0.2,
                        buffer_size=int(env.timeMax / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
                        actor=actor,
                        critic=critic,
                        path=simulationPath)
        '''重新加载Policy网络结构，这是必须的操作'''
        agent.PPO_info()

        if RETRAIN:
            agent.actor.load_state_dict(torch.load('Actor_PPO1950'))
            agent.critic.load_state_dict(torch.load('Critic_PPO1950'))
            # agent.agent_evaluate(50, True)

        max_training_timestep = int(env.timeMax / env.dt) * 20000  # 10000回合
        action_std_decay_freq = int(5e6)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

        sumr = 0
        start_eps = 0
        train_num = 0
        test_num = 0
        index = 0
        evaluate_r = np.array([])
        evaluate_e = np.array([])
        # while timestep <= max_training_timestep:
        while train_num < 10000:
            env.reset_random()
            # env.reset()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                # print(env.current_state)
                action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
                action_from_actor = action_from_actor.numpy()
                action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                # env.show_dynamic_image(isWait=False)  # 画图
                sumr += env.reward
                '''存数'''
                agent.buffer.append(s=env.current_state,
                                    a=action_from_actor,
                                    log_prob=a_log_prob.numpy(),
                                    r=env.reward,
                                    sv=s_value.numpy(),
                                    done=1.0 if env.is_terminal else 0.0,
                                    index=index)
                index += 1
                timestep += 1
                '''存数'''
                '''学习'''
                if timestep % agent.buffer.batch_size == 0:
                    print('========== LEARN ==========')
                    print('Episode: {}'.format(agent.episode))
                    print('Num of learning: {}'.format(train_num))
                    agent.learn()
                    '''clear buffer'''
                    average_train_r = round(sumr / (agent.episode + 1 - start_eps), 3)
                    print('Average reward:', average_train_r)
                    train_num += 1
                    start_eps = agent.episode
                    sumr = 0
                    index = 0
                    if train_num % 50 == 0 and train_num > 0:    # '1' should be 50
                        # rr, ee = agent.agent_evaluate(50, False)
                        # # print(rr)
                        # print('----- position errors -----')
                        # print('Training num:  ', train_num)
                        # print(ee)
                        # print('----- position errors -----')
                        # if train_num == 50:
                        #     evaluate_r = rr.copy()
                        #     evaluate_e = ee.copy()
                        # else:
                        #     evaluate_r = np.hstack((evaluate_r, rr))
                        #     evaluate_e = np.hstack((evaluate_e, ee))
                        # test_num += 1
                        # xx = np.arange(train_num - 50, train_num, 1)
                        # vis.line(X=xx, Y=rr, win='reward', update='append' if train_num > 50 else None, opts=dict(title='reward'))
                        # vis.line(X=xx, Y=ee, win='position error', update='append' if train_num > 50 else None, opts=dict(title='position error'))
                        print('check point save')
                        temp = simulationPath + 'training' + '_' + str(train_num) + '_save/'
                        os.mkdir(temp)
                        time.sleep(0.01)
                        torch.save(agent.actor.state_dict(), 'fuck.pkl')
                        # agent.actor.save_checkpoint(name='Actor_PPO', path=temp, num=train_num)
                        # agent.critic.save_checkpoint(name='Critic_PPO', path=temp, num=train_num)

                        ###############################################################################################3
                        # torch.save(agent.actor.state_dict(), 'fuck')
                        actor_temp = SoftmaxActor(3e-4, env.state_dim, env.action_dim, env.action_num)
                        actor_temp.load_state_dict(torch.load('fuck.pkl'))
                        # actor_temp.load_state_dict(agent.actor.state_dict())
                        # for p1, p2 in zip(agent.actor.parameters(), actor_temp.parameters()):
                        #     print(torch.linalg.norm(p1 - p2))
                        pts = np.array([
                            [0.5, 0.5], [0.5, 1.0], [0.5, 1.5], [0.5, 2.0], [0.5, 2.5], [0.5, 3.0], [0.5, 3.5], [0.5, 4.0], [0.5, 4.5],
                            [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5], [1.0, 3.0], [1.0, 3.5], [1.0, 4.0], [1.0, 4.5],
                            [1.5, 0.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [1.5, 2.5], [1.5, 3.0], [1.5, 3.5], [1.5, 4.0], [1.5, 4.5],
                            [2.0, 0.5], [2.0, 1.0], [2.0, 1.5], [2.0, 2.0], [2.0, 2.5], [2.0, 3.0], [2.0, 3.5], [2.0, 4.0], [2.0, 4.5],
                            [2.5, 0.5], [2.5, 1.0], [2.5, 1.5], [2.5, 2.0], [2.5, 2.5], [2.5, 3.0], [2.5, 3.5], [2.5, 4.0], [2.5, 4.5],
                            [3.0, 0.5], [3.0, 1.0], [3.0, 1.5], [3.0, 2.0], [3.0, 2.5]
                        ]).astype(np.float32)
                        rr2 = []
                        ee2 = []

                        for i in range(50):
                            print('------------------IIIII--------------------')
                            # self.env.reset_random()
                            env.init_target = pts[i]
                            env.reset()
                            r = 0
                            # self.env.reset()
                            while not env.is_terminal:
                                env.current_state = env.next_state.copy()
                                with torch.no_grad():
                                    t_state = torch.FloatTensor(env.current_state).to('cpu')
                                    _action_from_actor = actor_temp.evaluate(t_state)       # bad
                                    _action_from_actor2 = agent.actor.evaluate(t_state)     # good
                                    # for p1, p2 in zip(agent.actor.parameters(), actor_temp.parameters()):
                                    #     print('JJJJJ')
                                    #     print(torch.linalg.norm(p1-p2))
                                    # print('FUCK')
                                    # print(_action_from_actor, _action_from_actor2)
                                _action = agent.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
                                env.step_update(_action)  # 环境更新的action需要是物理的action
                                r += env.reward
                            print(np.linalg.norm(env.error))

                            env.init_target = pts[i]
                            env.reset()
                            while not env.is_terminal:
                                env.current_state = env.next_state.copy()
                                with torch.no_grad():
                                    t_state = torch.FloatTensor(env.current_state).to('cpu')
                                    _action_from_actor = agent.actor.evaluate(t_state)
                                _action = agent.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
                                env.step_update(_action)  # 环境更新的action需要是物理的action
                                r += env.reward
                                # env.show_dynamic_image(isWait=False)  # 画图
                            print(np.linalg.norm(env.error))
                            rr2.append(r)
                            ee2.append(np.linalg.norm(env.error))
                        # print(np.array(ee2))
                        ###############################################################################################3
                    print('========== LEARN ==========')
                '''学习'''

                # if timestep % action_std_decay_freq == 0:
                #     agent.decay_action_std(action_std_decay_rate, min_action_std)
            agent.episode += 1

    if TEST:
        actor = SoftmaxActor(3e-4, env.state_dim, env.action_dim, env.action_num, 'DiscreteActor', simulationPath)
        torch.save(actor.state_dict(), 'a.pkl')
        # for name, param in actor.named_parameters():
        #     print(name)
        #     print(param)
        actor1 = SoftmaxActor(3e-4, env.state_dim, env.action_dim, env.action_num, 'DiscreteActor', simulationPath)
        actor1.load_state_dict(torch.load('a.pkl'))
        # actor1.load_state_dict(actor.state_dict())
        # for name, param in actor1.named_parameters():
        #     print(name)
        #     print(param)
        for p1, p2 in zip(actor.parameters(), actor1.parameters()):
            print(torch.linalg.norm(p1-p2))
        # critic = Critic(3e-4, env.state_dim, env.action_dim, 'Critic', simulationPath)
        # agent = PPO_Dis(env=env,
        #                 gamma=0.99,
        #                 K_epochs=40,
        #                 eps_clip=0.2,
        #                 buffer_size=int(env.timeMax / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
        #                 actor=actor,
        #                 critic=critic,
        #                 path=simulationPath)
        # # agent.load_models(optPath + 'DPPO-4-CartPoleAngleOnly/')
        # agent.actor.load_state_dict(torch.load('Actor_PPO8400'))
        # rr, ee = agent.agent_evaluate(50, False)
        # print(rr)
        # print(ee)
