import sys
import datetime
import os
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from environment.envs.SecondOrderIntegration.SecondOrderIntegration import SecondOrderIntegration as env
from algorithm.value_base.DQN import DQN

from common.common_func import *
from common.common_cls import *

optPath = '../../datasave/network/DQN-SecondOrderIntegration/'
show_per = 50  # 每十个回合显示一次
is_storage_only_success = False
ALGORITHM = 'DQN'
ENV = 'DQN-SecondOrderIntegration'


class DQNNet(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, action_num=None, name='DQNNet', chkpt_dir=''):
        super(DQNNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        if action_num is None:
            self.action_num = env.action_num
        self.index = [0]
        for n in range(action_dim):
            self.index.append(self.index[n] + env.action_num[n])

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, sum(env.action_num))

        self.init()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def init(self):
        torch.nn.init.orthogonal_(self.fc1.weight, gain=1)
        torch.nn.init.uniform_(self.fc1.bias, 0, 1)
        torch.nn.init.orthogonal_(self.fc2.weight, gain=1)
        torch.nn.init.uniform_(self.fc2.bias, 0, 1)
        torch.nn.init.orthogonal_(self.out.weight, gain=1)
        torch.nn.init.uniform_(self.out.bias, 0, 1)

    def forward(self, _x):
        """
        :brief:         神经网络前向传播
        :param _x:      输入网络层的张量
        :return:        网络的输出
        """
        x = _x
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        state_action_value = self.out(x)

        return state_action_value


def fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file: str, randomEnv: bool, fullFillRatio: float,
                                                  epsilon: float, is_only_success: bool):
    """
    :brief:                     Full-fill the replay memory with current optimal policy
    :param torch_pkl_file:      ****.pkl, the neural network file
    :param randomEnv:           Initialize environment randomly or not
    :param fullFillRatio:       Percentage to fill up the replay memory
    :param epsilon:             exploration probability
    :param is_only_success:     only data leads to a stable episode can be added into replay memory
    :return:                    None
    """
    agent.target_net.load_state_dict(torch.load(torch_pkl_file))
    agent.eval_net.load_state_dict(torch.load(torch_pkl_file))
    env.reset_random() if randomEnv else env.reset()
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory_capacity)
    fullFillCount = max(min(fullFillCount, agent.memory_capacity), agent.batch_size)
    _new_state = []
    _new_action = []
    _new_reward = []
    _new_state_ = []
    _new_done = []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # 状态更新
            _numAction = agent.get_action_with_fixed_epsilon(env.current_state, epsilon)
            env.step_update(agent.actionNUm2PhysicalAction(_numAction))
            # env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1 if env.is_terminal else 0)
            else:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state,
                                              1 if env.is_terminal else 0)
                if agent.memory.mem_counter % 100 == 0:
                    print('replay_count = ', agent.memory.mem_counter)
        if is_only_success and env.terminal_flag == 3:
            agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
            print('replay_count = ', agent.memory.mem_counter)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :brief:                     Fulfill the replay memory with random policy
    :param randomEnv:           Initialize environment randomly or not
    :param fullFillRatio:       Percentage to fill up the replay memory
    :return:                    None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory_capacity)
    fullFillCount = max(min(fullFillCount, agent.memory_capacity), agent.batch_size)
    # state_episode, action_episode, reward_episode, next_state_episode, dones_episode = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        # state_episode.clear()
        # action_episode.clear()
        # reward_episode.clear()
        # next_state_episode.clear()
        # dones_episode.clear()
        # isOut = False
        while not env.is_terminal:
            # if agent.memory.mem_counter % 100 == 0:
            #     print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _numAction = agent.get_action_random()
            action = agent.actionNUm2PhysicalAction(_numAction)
            env.step_update(action)
            # env.show_dynamic_image(isWait=False)
            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
            # state_episode.append(env.current_state)
            # action_episode.append(env.current_action)
            # reward_episode.append(env.reward)
            # next_state_episode.append(env.next_state)
            # dones_episode.append(1 if env.is_terminal else 0)
            # isOut = isOut or env.is_out()
        # 只存储没有出界的episode
        # if not isOut:
        #     agent.memory.store_transition_per_episode(state_episode, action_episode, reward_episode, next_state_episode,
        #                                               dones_episode)
        #     print('replay_count = ', agent.memory.mem_counter)


if __name__ == '__main__':
    log_dir = '../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                          '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)

    TRAIN = False  # 直接训练
    RETRAIN = True  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    c = cv.waitKey(1)

    env = env(pos0=np.array([1.0, 1.0]),
              vel0=np.array([0.0, 0.0]),
              map_size=np.array([5.0, 5.0]),
              target=np.array([4.0, 4.0]),
              is_controller_BangBang=True)
    eval_net = DQNNet(state_dim=env.state_dim, action_dim=env.action_dim, action_num=env.action_num,
                      name='eval_net')  # action_num 是一个数组 !!
    target_net = DQNNet(state_dim=env.state_dim, action_dim=env.action_dim, action_num=env.action_num,
                        name='target_net')  # action_num 是一个数组 !!

    agent = DQN(env=env,
                gamma=0.9,
                epsilon=0.95,
                learning_rate=5e-4,
                memory_capacity=20000,  # 10000
                batch_size=512,
                target_replace_iter=100,
                eval_net=eval_net,
                target_net=target_net)

    if TRAIN:
        agent.DQN_info()
        # cv.waitKey(0)
        agent.save_episode.append(agent.episode)
        agent.save_reward.append(0.0)
        agent.save_epsilon.append(agent.epsilon)
        MAX_EPISODE = 600
        agent.episode = 0  # 设置起始回合
        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file='eval_parameters.pkl',
                                                          randomEnv=True,
                                                          fullFillRatio=0.5,
                                                          epsilon=0.5,
                                                          is_only_success=False)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.eval_net.init()
            # agent.target_net.init()
            '''生成初始数据之后要再次初始化网络'''
        else:
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
        print('Start to train...')
        new_state = []
        new_action = []
        new_reward = []
        new_state_ = []
        new_done = []
        while agent.episode <= MAX_EPISODE:
            # env.reset()
            env.reset_random()
            sumr = 0
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()
            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                agent.epsilon = agent.get_epsilon()
                # agent.epsilon = 0.4
                action_from_actor = agent.get_action_with_fixed_epsilon(env.current_state, agent.epsilon)
                action = agent.actionNUm2PhysicalAction(action_from_actor)
                env.step_update(action)  # 环境更新的action需要是物理的action
                if agent.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
                sumr = sumr + env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1 if env.is_terminal else 0)
                else:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state,
                                                  1 if env.is_terminal else 0)
                agent.learn(saveNNPath=simulationPath)
            '''跳出循环代表回合结束'''
            if is_storage_only_success and env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''跳出循环代表回合结束'''
            print(
                '=========START=========',
                'Episode:', agent.episode,
                'Epsilon:', agent.epsilon,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========')
            print()
            agent.saveData_EpisodeRewardEpsilon(agent.episode, sumr, agent.epsilon)
            agent.episode += 1
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        agent.saveData_EpisodeRewardEpsilon(0.0, 0.0, 0.0, True, 'EpisodeRewardEpsilon.csv', simulationPath)
        agent.saveData_StepTDErrorNNLose(0.0, 0.0, 0.0, True, 'StepTDErrorNNLose.csv', simulationPath)
        '''dataSave'''

    else:
        print('TESTing...')
        agent.get_optimalfrompkl(optPath + 'dqn-4-second-order-integration.pkl')
        # cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
        #                      cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
        #                      120.0,
        #                      (env.width, env.height))
        simulation_num = 10
        error = []
        terminal_list = []
        for i in range(simulation_num):
            print('==========START==========')
            print('episode = ', i)
            env.reset_random()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                env.current_state = env.next_state.copy()
                env.step_update(
                    agent.actionNUm2PhysicalAction(agent.get_action_with_fixed_epsilon(env.current_state, 0.0)))
                env.show_dynamic_image(isWait=False)
                # cap.write(env.save)
            error.append(np.linalg.norm(env.error))
            terminal_list.append(env.init_target)
            print('===========END===========')

        '''统计一下，没有什么特殊的'''
        error = np.array(error)
        terminal_list = np.array(terminal_list)
        norm_error = (error - np.min(error)) / (np.max(error) - np.min(error))
        color = []
        for _e in norm_error:
            color.append((_e, 0., 0.))
        print('Mean error  ', error.mean())
        print('Std error   ', error.std())
        print('Max error   ', np.max(error))
        print('Min error   ', np.min(error))
        plt.figure(0)
        plt.plot(range(simulation_num), error)
        plt.ylim(0, 0.35)
        plt.yticks(np.arange(0, 0.35, 0.05))

        plt.figure(1)
        plt.hist(error, rwidth=0.05)

        print('分布图')
        plt.figure(2)
        plt.scatter(x=terminal_list[:, 0], y=terminal_list[:, 1], marker='o', c=color,
                    s=[25 for _ in range(simulation_num)])  #
        plt.axis('equal')
        plt.xlim(0, env.map_size[0])
        plt.ylim(0, env.map_size[1])
        plt.xticks(np.arange(0, 1, env.map_size[0]))
        plt.yticks(np.arange(0, 1, env.map_size[1]))
        plt.show()

        cv.destroyAllWindows()
