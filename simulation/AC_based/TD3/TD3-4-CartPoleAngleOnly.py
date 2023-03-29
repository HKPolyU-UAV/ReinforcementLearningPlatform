import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
# import copy
from environment.envs.cartpole.cartpole_angleonly import CartPoleAngleOnly
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from common.common_func import *
from common.common_cls import *


optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ALGORITHM = 'TD3'
ENV = 'CartPoleAngleOnly'


class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_td3'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_td3ALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        # self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(_action))
        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()

        self.action_value.reset_parameters()
        self.q.reset_parameters()

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

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
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_td3'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_td3ALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # 输入 -> 第一个隐藏层
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 128)  # 第一个隐藏层 -> 第二个隐藏层
        self.batch_norm2 = nn.LayerNorm(128)

        # self.fc3 = nn.Linear(64, 32)  # 第2个隐藏层 -> 第3个隐藏层
        # self.batch_norm3 = nn.LayerNorm(32)

        self.mu = nn.Linear(128, self.action_dim)  # 第3个隐藏层 -> 输出层

        # self.initialization()
        self.initialization_default()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()
        # self.fc3.reset_parameters()
        # self.batch_norm3.reset_parameters()
        self.mu.reset_parameters()

    def forward(self, state):
        x = self.fc1(state)
        x = self.batch_norm1(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)

        # x = self.fc3(x)
        # x = self.batch_norm3(x)
        # x = func.relu(x)

        x = torch.tanh(self.mu(x))  # bound the output to [-1, 1]

        return x

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
    pass


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
        # env.reset_random() if randomEnv else env.reset()
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
                if agent.memory.mem_counter % 1000 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
                if env.reward >= -10:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            '''设置一个限制，只有满足某些条件的[s a r s' done]才可以被加进去'''
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
                print('replay_count = ', agent.memory.mem_counter)


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

    env = CartPoleAngleOnly(initTheta=deg2rad(0), save_cfg=False)

    if TRAIN:
        actor = Actor(1e-4, env.state_dim, env.action_dim, 'Actor', simulationPath)
        target_actor = Actor(1e-4, env.state_dim, env.action_dim, 'TargetActor', simulationPath)
        critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic1', simulationPath)
        target_critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic1', simulationPath)
        critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic2', simulationPath)
        target_critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic2', simulationPath)
        agent = TD3(env=env,
                    gamma=0.99, noise_clip=1 / 2, noise_policy=1 / 4, policy_delay=3,
                    critic1_soft_update=1e-2,
                    critic2_soft_update=1e-2,
                    actor_soft_update=1e-2,
                    memory_capacity=50000,  # 100000
                    batch_size=512,  # 1024
                    actor=actor,
                    target_actor=target_actor,
                    critic1=critic1,
                    target_critic1=target_critic1,
                    critic2=critic2,
                    target_critic2=target_critic2,
                    path=simulationPath)

        agent.TD3_info()
        successCounter = 0
        timeOutCounter = 0
        collisionCounter = 0
        # cv.waitKey(0)
        MAX_EPISODE = 20000

        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.5, is_only_success=is_storage_only_success)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.actor.initialization_default()
            # agent.target_actor.initialization_default()
            # agent.critic.initialization_default()
            # agent.target_critic.initialization_default()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5, is_only_success=is_storage_only_success)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        step = 0
        while agent.episode <= MAX_EPISODE:
            print('=========START=========')
            print('Episode:', agent.episode)
            env.reset_random()
            sumr = 0
            while not env.is_terminal:      # 每个回合
                timestep += 1
                env.current_state = env.next_state.copy()
                epsilon = random.uniform(0, 1)
                if epsilon < 0.15:
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 4)  # 剩下的是神经网络加噪声
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                step += 1
                env.show_dynamic_image(isWait=False)       # 画图
                sumr = sumr + env.reward
                if env.reward >= -10:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)

                agent.learn(is_reward_ascent=False)

            print('Cumulative reward:', round(sumr, 3))
            print('TimeStep:', timestep)
            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()
                agent.agent_evaluate(5)
            if timestep % 500 == 0:
                print('check point save')
                temp = simulationPath + 'timestep' + '_' + str(timestep) + '_save/'
                os.mkdir(temp)
                time.sleep(0.01)
                agent.actor.save_checkpoint(name='Actor_td3', path=temp, num=timestep)
                agent.target_actor.save_checkpoint(name='TargetActor_td3', path=temp, num=timestep)
                agent.critic1.save_checkpoint(name='Critic1_td3', path=temp, num=timestep)
                agent.target_critic1.save_checkpoint(name='TargetCritic1_td3', path=temp, num=timestep)
                agent.critic2.save_checkpoint(name='Critic2_td3', path=temp, num=timestep)
                agent.target_critic2.save_checkpoint(name='TargetCritic2_td3', path=temp, num=timestep)

    if TEST:
        agent = TD3(env=env, target_actor=Actor(1e-4, env.state_dim, env.action_dim, 'TargetActor', simulationPath))
        agent.load_target_actor_optimal(path=optPath, file='TD3-CartPoleAngleOnly/parameters/TargetActor_td3')

        for _ in range(3):
            env.reset_random()
            while not env.is_terminal:
                cv.waitKey(1)
                env.current_state = env.next_state.copy()
                t_state = torch.tensor(env.current_state, dtype=torch.float).to(agent.target_actor.device)
                mu = agent.target_actor(t_state).to(agent.target_actor.device)
                action = agent.action_linear_trans(mu.cpu().detach().numpy())
                env.step_update(action)
                if agent.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
        cv.destroyAllWindows()
