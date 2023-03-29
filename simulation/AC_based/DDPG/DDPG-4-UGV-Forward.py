import os
import sys
import datetime
import pandas as pd
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

# import copy
from environment.envs.UGV.ugv_forward_continuous import UGV_Forward_Continuous
from algorithm.actor_critic.DDPG import DDPG
from common.common_func import *
from common.common_cls import *

cfgPath = '../../../environment/config/'
cfgFile = 'UGV_Forward_Continuous.xml'
show_per = 1
ALGORITHM = 'DDPG'
ENV = 'UGV-ForwardOnly'


class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        # self.initialization()
        self.initialization_default()

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

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()
        self.action_value.reset_parameters()
        self.q.reset_parameters()

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
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

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
            _action_from_actor = agent.choose_action(env.current_state, is_optimal=False, sigma=1 / 3)
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
            # _action_from_actor = agent.choose_action(env.current_state, False, sigma=1.0)
            _action_from_actor = agent.choose_action_random()
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
    TRAIN = False  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN
    is_storage_only_success = False

    env = UGV_Forward_Continuous(initPhi=deg2rad(0), save_cfg=False, x_size=5.0, y_size=5.0, start=[0.3, 0.3], terminal=[0.8, 0.8])

    if TRAIN:
        actor = Actor(5e-5, env.state_dim, env.action_dim, 'Actor', simulationPath)
        target_actor = Actor(5e-5, env.state_dim, env.action_dim, 'TargetActor', simulationPath)
        critic = Critic(5e-4, env.state_dim, env.action_dim, 'Critic', simulationPath)
        target_critic = Critic(5e-4, env.state_dim, env.action_dim, 'TargetCritic', simulationPath)
        agent = DDPG(env=env,
                     gamma=0.99,  # 0.99
                     actor_soft_update=5e-3,  # 1e-2
                     critic_soft_update=5e-3,  # 1e-2
                     memory_capacity=60000,  # 60000
                     batch_size=512,  # 512
                     actor=actor,
                     target_actor=target_actor,
                     critic=critic,
                     target_critic=target_critic,
                     path=simulationPath)
        agent.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
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
            # agent.target_actor.initialization()
            # agent.critic.initialization()
            # agent.target_critic.initialization()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5, is_only_success=is_storage_only_success)
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
            sigma = 0.5 * math.exp(-0.005181 * agent.episode) + 0.5 * math.exp(-0.0007913 * agent.episode)
            # sigma = 1 / 3

            # epsilon = 1.212e-6 * agent.episode ** 2 - 0.001697 * agent.episode + 0.7848
            epsilon = 0.1
            # if agent.episode % 10 == 0:     # 每10个回合将经验池重新排序一次
            #     agent.memory.get_reward_sort()
            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                if random.uniform(0, 1) < epsilon:
                    # print('...random...')
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=sigma)  # 剩下的是神经网络加噪声
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                if agent.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
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
            if env.terminal_flag == 3:
                successCounter += 1
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if is_storage_only_success:  # 只用超时的或者成功的训练
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                    # agent.memory.get_reward_resort(per=5)
            '''跳出循环代表回合结束'''
            print('Cumulative reward:', round(sumr, 3), 'Sigma: ', round(sigma, 4), 'epsilon: ', round(epsilon, 3))
            print('共：', agent.episode, ' 回合，成功', successCounter, ' 回合，超时', timeOutCounter, ' 回合')
            if agent.episode > 0:
                print('成功率：', round(successCounter / agent.episode, 3))
            print('==========END=========')
            print()
            agent.saveData_EpisodeReward(episode=agent.episode,
                                         time=env.time,
                                         reward=sumr,
                                         average_reward=sumr / env.time,
                                         successrate=round(successCounter / max(agent.episode, 1), 3),
                                         is2file=False, filename='EpisodeReward.csv')
            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()
            if agent.episode % 100 == 0:
                print('check point save')
                agent.actor.save_checkpoint(name='Actor_', path=simulationPath, num=agent.episode)
                agent.saveData_EpisodeReward(episode=0, time=1.0, reward=0.0, average_reward=0.0, successrate=1.0, is2file=True, filename='EpisodeReward.csv')
                agent.saveData_Step_Reward(step=0, reward=0, is2file=True, filename='StepReward.csv')
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        agent.save_models_all()
        agent.saveData_EpisodeReward(episode=0, time=1.0, reward=0.0, average_reward=0.0, successrate=1.0, is2file=True, filename='EpisodeReward.csv')
        agent.saveData_Step_Reward(step=0, reward=0, is2file=True, filename='StepReward.csv')
        '''dataSave'''

    if TEST:
        print('TESTing...')
        optPath = '../../../datasave/network/DDPG-UGV-Forward/parameters/'
        agent = DDPG(env=env, target_actor=Actor(1e-4, env.state_dim, env.action_dim, 'Actor', simulationPath))
        agent.load_target_actor_optimal(path=optPath, file='TargetActor_ddpg')
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (env.width, env.height))
        simulation_num = 10
        successCounter = 0
        timeOutCounter = 0
        failStartx, failStarty = [], []
        failTerminalx, failTerminaly = [], []
        for i in range(simulation_num):
            print('==========START' + str(i) + '==========')
            env.reset_random()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                env.current_state = env.next_state.copy()
                action_from_actor = agent.evaluate(env.current_state)
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                currentError = dis_two_points([env.x, env.y], env.terminal)
                env.step_update(action)
                nextError = dis_two_points([env.x, env.y], env.terminal)
                env.show_dynamic_image(isWait=False)
                cap.write(env.save)
                env.saveData(is2file=False)
            print('===========END===========')
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if env.terminal_flag == 3:
                successCounter += 1
        print('Total:', simulation_num, '  successful:', successCounter, '  timeout:', timeOutCounter)
        print('success rate:', round(successCounter / simulation_num))
        saveData = pd.DataFrame({'sx': failStartx,
                                 'sy': failStarty,
                                 'tx': failTerminalx,
                                 'ty': failTerminaly})
        saveData.to_csv(simulationPath + 'faildata.csv')
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)
        # agent.save_models_all()
