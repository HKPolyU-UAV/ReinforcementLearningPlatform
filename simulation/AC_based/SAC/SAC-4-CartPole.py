import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

# import copy
from environment.envs.cartpole.cartpole import CartPole
from algorithm.actor_critic.Soft_Actor_Critic import Soft_Actor_Critic as SAC
from environment.envs.PIDControl.pid import PID
from common.common_cls import *
from common.common_func import *


cfgPath = '../../../environment/config/'
cfgFile = 'cartpole.xml'
show_per = 1
ALGORITHM = 'TD3'
ENV = 'CartPole'


class DualCritic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(DualCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_sac'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_sacALL'

        self.L11 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.batch_norm11 = nn.LayerNorm(256)
        self.L12 = nn.Linear(256, 256)
        self.batch_norm12 = nn.LayerNorm(256)
        self.L13 = nn.Linear(256, 1)

        self.L21 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.batch_norm21 = nn.LayerNorm(256)
        self.L22 = nn.Linear(256, 256)
        self.batch_norm22 = nn.LayerNorm(256)
        self.L23 = nn.Linear(256, 1)

        self.initialization_default()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        sa = torch.cat([state, _action], 1)
        # sa_value1 = self.L11(sa)
        # sa_value1 = self.batch_norm11(sa_value1)
        # sa_value1 = func.relu(sa_value1)
        # sa_value1 = self.L12(sa_value1)
        # sa_value1 = self.batch_norm12(sa_value1)
        # sa_value1 = func.relu(sa_value1)
        # sa_value1 = self.L13(sa_value1)
        sa_value1 = func.relu(self.batch_norm11(self.L11(sa)))
        sa_value1 = func.relu(self.batch_norm12(self.L12(sa_value1)))
        sa_value1 = self.L13(sa_value1)

        # sa_value2 = self.L11(sa)
        # sa_value2 = self.batch_norm11(sa_value2)
        # sa_value2 = func.relu(sa_value2)
        # sa_value2 = self.L12(sa_value2)
        # sa_value2 = self.batch_norm12(sa_value2)
        # sa_value2 = func.relu(sa_value2)
        # sa_value2 = self.L13(sa_value2)
        sa_value2 = func.relu(self.batch_norm21(self.L21(sa)))
        sa_value2 = func.relu(self.batch_norm22(self.L22(sa_value2)))
        sa_value2 = self.L23(sa_value2)

        return sa_value1, sa_value2

    def initialization_default(self):
        self.L11.reset_parameters()
        self.batch_norm11.reset_parameters()
        self.L12.reset_parameters()
        self.batch_norm12.reset_parameters()
        self.L13.reset_parameters()

        self.L21.reset_parameters()
        self.batch_norm21.reset_parameters()
        self.L22.reset_parameters()
        self.batch_norm22.reset_parameters()
        self.L23.reset_parameters()

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
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(ProbActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_sac'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_sacALL'

        self.l1 = nn.Linear(self.state_dim, 128)  # 输入 -> 第一个隐藏层
        self.batch_norm1 = nn.LayerNorm(128)

        self.l2 = nn.Linear(128, 128)  # 第一个隐藏层 -> 第二个隐藏层
        self.batch_norm2 = nn.LayerNorm(128)

        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

        self.initialization_default()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization_default(self):
        self.l1.reset_parameters()
        self.batch_norm1.reset_parameters()

        self.l2.reset_parameters()
        self.batch_norm2.reset_parameters()

        self.mean_layer.reset_parameters()
        self.log_std_layer.reset_parameters()

    def forward(self, state, deterministic=False):
        x = self.l1(state)
        x = self.batch_norm1(x)
        x = func.relu(x)

        x = self.l2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)

        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.exp(log_std)

        gaussian_dist = Normal(mean, std)
        if deterministic:
            act = mean
        else:
            act = gaussian_dist.rsample()

        # The method refers to Open AI Spinning up, which is more stable.
        log_pi = gaussian_dist.log_prob(act).sum(dim=1, keepdim=True)
        log_pi -= (2 * (np.log(2) - act - func.softplus(-2 * act))).sum(dim=1, keepdim=True)

        act = torch.tanh(act)

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


def fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    agent.load_models(path='../../../datasave/network/DDPG-UGV-Forward/parameters/')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = agent.choose_action(env.current_state, False)
            _action = agent.action_linear_trans(_action_from_actor)
            env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                if agent.memory.mem_counter % 100 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
                '''设置一个限制，只有满足某些条件的[s a r s' done]才可以被加进去'''
                if env.reward >= -3.5:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                print('replay_count = ', agent.memory.mem_counter)
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
    agent.memory.get_reward_sort()


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    """
    :param randomEnv:           init env randomly
    :param fullFillRatio:       the ratio
    :param is_only_success:
    :return:
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = agent.choose_action_random()
            _action = agent.action_linear_trans(_action_from_actor)
            env.step_update(_action)
            # env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                if agent.memory.mem_counter % 500 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
                '''设置一个限制，只有满足某些条件的[s a r s' done]才可以被加进去'''
                # if env.reward >= -4.5:
                if True:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
                print('replay_count = ', agent.memory.mem_counter)
    agent.memory.get_reward_sort()


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

    env = CartPole(initTheta=0, initX=0, save_cfg=False)

    if TRAIN:
        actor = ProbActor(3e-4, env.state_dim, env.action_dim, 'Actor', simulationPath)
        critic = DualCritic(5e-4, env.state_dim, env.action_dim, 'Critic', simulationPath)
        target_critic = DualCritic(5e-4, env.state_dim, env.action_dim, 'TargetCritic', simulationPath)
        critic.load_state_dict(target_critic.state_dict())  # 二者初始化参数必须相同
        for p in target_critic.parameters():  # target 网络不训练
            p.requires_grad = False
        agent = SAC(env=env,
                    gamma=0.99,  # 0.99
                    actor_soft_update=5e-3,  # 1e-2
                    critic_soft_update=5e-3,  # 1e-2
                    alpha=0.2,
                    alpha_lr=3e-4,
                    alpha_learning=True,
                    memory_capacity=60000,       #
                    batch_size=512,
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    path=simulationPath)
        agent.SAC_info()
        # cv.waitKey(0)
        MAX_EPISODE = 5000
        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True,
                                              fullFillRatio=0.5,
                                              is_only_success=is_storage_only_success)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.actor.initialization()
            # agent.critic.initialization()
            # agent.target_critic.initialization()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.25, is_only_success=is_storage_only_success)
            '''fullFillReplayMemory_Random'''

        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        while agent.episode <= MAX_EPISODE:
            print('=========START ' + str(agent.episode) + '=========')
            env.reset_random()
            sumr = 0
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()

            epsilon = 0.1

            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                if random.uniform(0, 1) < epsilon:
                    # print('...random...')
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False)
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                # agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                # if agent.episode % show_per == 0:
                #     env.show_dynamic_image(isWait=False)
                sumr += env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    '''设置一个限制，只有满足某些条件的[s a r s' done]才可以被加进去'''
                    # if env.reward >= -4.5:
                    if True:
                        agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
            '''跳出循环代表一个回合结束'''
            if is_storage_only_success:  # 只用超时的或者成功的训练
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                    # agent.memory.get_reward_resort(per=5)
            '''跳出循环代表回合结束'''
            print('Cumulative reward:', round(sumr, 3))
            print('==========END=========')
            print()

            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()

            if agent.episode % 50 == 0:
                '''50个回合测试一下'''
                agent.agent_evaluate(test_num=5)
                '''50个回合测试一下'''

            if agent.episode % 100 == 0:
                print('check point save')
                agent.actor.save_checkpoint(name='Actor_', path=simulationPath, num=agent.episode)

            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        agent.save_models_all()
        '''dataSave'''

    if TEST:
        pass
