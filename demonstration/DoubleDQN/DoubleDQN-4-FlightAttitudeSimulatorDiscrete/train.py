import sys
import datetime
import os
import cv2 as cv
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from FlightAttitudeSimulatorDiscrete import FlightAttitudeSimulatorDiscrete as env
from algorithm.value_base.Double_DQN import Double_DQN

is_storage_only_success = False
ALGORITHM = 'DQN'
ENV = 'FlightAttitudeSimulatorDiscrete'


class DQNNet(nn.Module):
    def __init__(self, state_dim=1, action_dim=1):
        super(DQNNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

        self.init()

        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_action_value = self.out(x)

        return state_action_value


def fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file: str, randomEnv: bool, fullFillRatio: float, epsilon: float, is_only_success: bool):
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
            env.visualization()
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
    print('Collecting...')
    print(agent.memory_capacity)
    fullFillCount = int(fullFillRatio * agent.memory_capacity)
    fullFillCount = max(min(fullFillCount, agent.memory_capacity), agent.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        if env.is_terminal:
            env.reset_random() if randomEnv else env.reset()
        else:
            if agent.memory.mem_counter % 1000 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _numAction = agent.get_action_random()
            action = agent.actionNUm2PhysicalAction(_numAction)
            env.step_update(action)
            # env.visualization()
            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


def get_epsilon():
    """
	:brief:             get the exploration probability of an episode
	:return:            episode
	"""
    # self.epsilon = 0.2       # It is a user-defined module.
    if 0 <= agent.episode <= 300:
        epsilon = 2.222e-06 * agent.episode ** 2 - 0.001667 * agent.episode + 0.9  # FAS
    elif 300 < agent.episode <= 600:
        epsilon = 2.222e-06 * agent.episode ** 2 - 0.003 * agent.episode + 1.45  # FAS
    elif 600 < agent.episode <= 900:
        epsilon = 2.222e-06 * agent.episode ** 2 - 0.004333 * agent.episode + 2.4  # FAS
    elif 900 < agent.episode <= 1200:
        epsilon = 2.222e-06 * agent.episode ** 2 - 0.005667 * agent.episode + 3.75  # FAS
    else:
        epsilon = 0.1
    return epsilon


if __name__ == '__main__':
    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)

    RETRAIN = False  # 基于之前的训练结果重新训练

    c = cv.waitKey(1)

    env = env()
    eval_net = DQNNet(state_dim=env.state_dim, action_dim=env.action_num[0])
    target_net = DQNNet(state_dim=env.state_dim, action_dim=env.action_num[0])

    agent = Double_DQN(env=env,
                gamma=0.99,
                epsilon=0.95,
                learning_rate=1e-4,
                memory_capacity=5000,  # 10000
                batch_size=512,
                target_replace_iter=50,
                eval_net=eval_net,
                target_net=target_net)

    agent.DQN_info()
    # cv.waitKey(0)
    agent.save_episode.append(agent.episode)
    agent.save_reward.append(0.0)
    agent.save_epsilon.append(agent.epsilon)
    MAX_EPISODE = 1500
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
            agent.epsilon = get_epsilon()
            # agent.epsilon = 0.4
            action_from_actor = agent.get_action_with_fixed_epsilon(env.current_state, agent.epsilon)
            action = agent.actionNUm2PhysicalAction(action_from_actor)
            env.step_update(action)  # 环境更新的action需要是物理的action
            if agent.episode % 10 == 0:
                env.visualization()

            sumr += env.reward
            if is_storage_only_success:
                new_state.append(env.current_state)
                new_action.append(env.current_action)
                new_reward.append(env.reward)
                new_state_.append(env.next_state)
                new_done.append(1 if env.is_terminal else 0)
            else:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)

            agent.lean_double_dqn(path=simulationPath, is_reward_ascent=False, item=1)
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
        agent.episode += 1
