import sys
import datetime
import os
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from environment.envs.FlightAttitudeSimulator.flight_attitude_simulator import Flight_Attitude_Simulator as flight_sim
from algorithm.value_base.DQN import DQN

from common.common_func import *
from common.common_cls import *

optPath = '../../datasave/network/'
show_per = 1  # 每十个回合显示一次
is_storage_only_success = False
ALGORITHM = 'DQN'
ENV = 'FlightAttitudeSimulator'


class DQNNet(nn.Module):
    def __init__(self, _input: int, _output: list):
        """
        :brief:             神经网络初始化
        :param _input:      输入维度
        :param _output:     输出维度
        """
        super(DQNNet, self).__init__()
        self.hidden1 = nn.Linear(_input, 64)  # input -> hidden1
        self.hidden2 = nn.Linear(64, 64)  # hidden1 -> hidden2
        self.out1 = nn.Linear(64, _output[0])  # hidden2 -> output
        self.init()

    def init(self):
        torch.nn.init.orthogonal_(self.hidden1.weight, gain=1)
        torch.nn.init.uniform_(self.hidden1.bias, 0, 1)
        torch.nn.init.orthogonal_(self.hidden2.weight, gain=1)
        torch.nn.init.uniform_(self.hidden2.bias, 0, 1)
        torch.nn.init.orthogonal_(self.out.weight, gain=1)
        torch.nn.init.uniform_(self.out1.bias, 0, 1)

    def forward(self, _x):
        """
        :brief:         神经网络前向传播
        :param _x:      输入网络层的张量
        :return:        网络的输出
        """
        x = _x
        x = self.hidden1(x)
        x = func.relu(x)
        x = self.hidden2(x)
        x = func.relu(x)
        state_action_value = self.out(x)
        return state_action_value


def fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file: str, randomEnv: bool, fullFillRatio: float, epsilon: float, is_only_success: bool):
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
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(agent.actionNUm2PhysicalAction(_numAction))
            env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1 if env.is_terminal else 0)
            else:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                if agent.memory.mem_counter % 100 == 0:
                    print('replay_count = ', agent.memory.mem_counter)
        if is_only_success and env.terminal_flag == 3:
            agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
            print('replay_count = ', agent.memory.mem_counter)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :brief:                     Full fill the replay memory with random policy
    :param randomEnv:           Initialize environment randomly or not
    :param fullFillRatio:       Percentage to fill up the replay memory
    :return:                    None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory_capacity)
    fullFillCount = max(min(fullFillCount, agent.memory_capacity), agent.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _numAction = agent.get_action_random()
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(agent.actionNUm2PhysicalAction(_numAction))
            env.show_dynamic_image(isWait=False)
            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
    log_dir = '../../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)

    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    c = cv.waitKey(1)

    env = flight_sim(initTheta=-60.0, setTheta=0.0, save_cfg=False)
    eval_net = DQNNet(_input=env.state_dim, _output=env.action_num)      # action_num 是一个数组 !!
    target_net = DQNNet(_input=env.state_dim, _output=env.action_num)  # action_num 是一个数组 !!

    agent = DQN(env=env,
                gamma=0.9,
                epsilon=0.95,
                learning_rate=5e-4,
                memory_capacity=20000,  # 10000
                batch_size=256,
                target_replace_iter=200,
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
            fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file='dqn_parameters_ok3.pkl',
                                                          randomEnv=True,
                                                          fullFillRatio=0.5,
                                                          epsilon=0.5,
                                                          is_only_success=True)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.eval_net.init()
            # agent.target_net.init()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
            '''fullFillReplayMemory_Random'''
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
                # dqn.epsilon = dqn.get_epsilon()
                agent.epsilon = 0.4
                numAction = agent.get_action_with_fixed_epsilon(env.current_state, agent.epsilon)
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = \
                    env.step_update(agent.actionNUm2PhysicalAction(numAction))  # 环境更新的action需要是物理的action
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
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.nn_training(saveNNPath=simulationPath)
            '''跳出循环代表回合结束'''
            if is_storage_only_success and env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''跳出循环代表回合结束'''
            print(
                '=========START=========',
                'Episode:', agent.episode,
                'Epsilon', agent.epsilon,
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

    if TEST:
        print('TESTing...')
        agent.get_optimalfrompkl(optPath + 'dqn-4-flight-attitude-simulator.pkl')
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
                             cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             120.0,
                             (env.width, env.height))
        simulation_num = 5
        for i in range(simulation_num):
            print('==========START==========')
            print('episode = ', i)
            env.reset_random()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                env.current_state = env.next_state.copy()
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = \
                    env.step_update(agent.actionNUm2PhysicalAction(agent.get_action_with_fixed_epsilon(env.current_state, 0.0)))
                env.show_dynamic_image(isWait=False)
                cap.write(env.save)
                env.saveData(is2file=False)
            print('Stable Theta:', rad2deg(env.theta), '\t', 'Stable error:', rad2deg(env.setTheta - env.theta))
            print('===========END===========')
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)
