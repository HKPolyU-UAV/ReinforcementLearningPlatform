import sys
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UGV import UGV as env


class SACActor(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, a_min: np.ndarray = np.zeros(3),
                 a_max: np.ndarray = np.ones(3)):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        self.device = 'cpu'
        self.a_min = torch.tensor(a_min, dtype=torch.float).to(self.device)
        self.a_max = torch.tensor(a_max, dtype=torch.float).to(self.device)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.to(self.device)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - func.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = torch.tanh(a) * self.gain + self.off

        return a, log_pi

    def evaluate(self, s):
        s_t = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _ = self.forward(s_t, deterministic=True)
        return a.data.numpy().flatten()


if __name__ == '__main__':
    optPath = './datasave/net/'
    env = env()
    eval_net = SACActor(state_dim=env.state_dim, action_dim=env.action_dim, a_min=env.action_range[:, 0],
                        a_max=env.action_range[:, 1])
    eval_net.load_state_dict(torch.load(optPath + 'actor'))

    n = 10
    for _ in range(n):
        env.reset(True)
        sumr = 0

        while not env.is_terminal:
            c = cv.waitKey(1)
            env.current_state = env.next_state.copy()
            action_from_actor = eval_net.evaluate(env.current_state)
            env.step_update(action_from_actor)
            env.visualization()

            sumr += env.reward
        print('Cumulative reward:', round(sumr, 3))
        print()
