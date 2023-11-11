import os
import sys
import torch.nn as nn
import torch
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from cartpole_angleonly import CartPoleAngleOnly as env
from utils.functions import *

test_episode = []
test_reward = []
sumr_list = []


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 state_dim: int = 3,
                 action_dim: int = 3,
                 a_min: np.ndarray = np.zeros(3),
                 a_max: np.ndarray = np.ones(3),
                 init_std: float = 0.5,
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.activate_func = nn.Tanh()
        self.a_min = torch.tensor(a_min, dtype=torch.float)
        self.a_max = torch.tensor(a_max, dtype=torch.float)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.action_dim = action_dim
        self.std = torch.tensor(init_std, dtype=torch.float)

        if use_orthogonal_init:
            self.orthogonal_init_all()

    @staticmethod
    def orthogonal_init(layer, gain=1.0):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

    def orthogonal_init_all(self):
        self.orthogonal_init(self.fc1)
        self.orthogonal_init(self.fc2)
        self.orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
        # mean = torch.relu(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action_mean = self.forward(t_state)
        return action_mean.detach().cpu().numpy().flatten()


if __name__ == '__main__':
    env = env(0.)
    opt_actor = PPOActor_Gaussian(state_dim=env.state_dim,
                                  action_dim=env.action_dim,
                                  a_min=np.array(env.action_range)[:, 0],
                                  a_max=np.array(env.action_range)[:, 1],
                                  init_std=0.01,
                                  use_orthogonal_init=True)
    optPath = os.path.dirname(os.path.abspath(__file__)) + '/datasave/net/'
    opt_actor.load_state_dict(torch.load(optPath + 'actor'))  # 测试时，填入测试actor网络

    n = 10
    for i in range(10):
        env.reset_random()
        test_r = 0.
        while not env.is_terminal:
            a = opt_actor.evaluate(env.current_state)
            env.step_update(a)
            test_r += env.reward
            env.visualization()
        print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
