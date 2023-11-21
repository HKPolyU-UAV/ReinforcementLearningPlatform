import sys
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from SecondOrderIntegration import SecondOrderIntegration as env

is_storage_only_success = False
ALGORITHM = 'DQN'
ENV = 'SecondOrderIntegration'


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

    def evaluate(self, s):
        t_state = torch.tensor(s).float().to(self.device)
        v = self.forward(t_state).cpu().detach().numpy()
        return np.argmax(v)


if __name__ == '__main__':
    optPath = './datasave/net/'
    env = env()
    eval_net = DQNNet(state_dim=env.state_dim, action_dim=env.action_num[0])
    eval_net.load_state_dict(torch.load(optPath + 'eval'))
    # video = cv.VideoWriter('../DQN-4-' + env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 200,
    #                        (env.image_size[0], env.image_size[1]))
    n = 5

    for _ in range(n):
        env.reset(random=True)
        sumr = 0

        while not env.is_terminal:
            c = cv.waitKey(1)
            env.current_state = env.next_state.copy()
            action_from_actor = eval_net.evaluate(env.current_state)
            action = np.array([env.action_space[0][action_from_actor]])
            env.step_update(action)
            env.visualization()
            # video.write(env.image)
            sumr += env.reward
        print('Cumulative reward:', round(sumr, 3))
        print()
    # video.release()
