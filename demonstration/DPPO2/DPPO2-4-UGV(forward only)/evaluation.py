import os
import sys
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from UGV import UGV
from utils.functions import *

timestep = 0
ENV = 'UGV(forward only)'
ALGORITHM = 'PPO2'
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
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
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
        self.orthogonal_init(self.net[0])
        self.orthogonal_init(self.net[2])
        self.orthogonal_init(self.net[4], gain=0.01)

    def forward(self):
        raise NotImplementedError

    def get_dist(self, s):
        mean = self.net(s)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action_mean = self.net(t_state)
        return action_mean.detach().cpu().numpy().flatten()


if __name__ == '__main__':
    env = UGV()
    env_msg = {'state_dim': env.state_dim, 'action_dim': env.action_dim, 'name': env.name, 'action_range': env.action_range}
    t_epoch = 0  # 当前训练次数
    test_num = 0

    optPath = './datasave/net/'
    # optPath = './trainNum_22500/'
    opt_actor = PPOActor_Gaussian(state_dim=env.state_dim,
                                  action_dim=env.action_dim,
                                  a_min=np.array(env.action_range)[:, 0],
                                  a_max=np.array(env.action_range)[:, 1],
                                  init_std=1.2,
                                  use_orthogonal_init=True)
    opt_actor.load_state_dict(torch.load(optPath + 'actor'))

    n = 10
    for i in range(n):
        env.reset(random=True)
        test_r = 0.
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            _a = opt_actor.evaluate(env.current_state)
            env.step_update(_a)
            test_r += env.reward
            env.visualization()
        test_num += 1
        test_reward.append(test_r)
