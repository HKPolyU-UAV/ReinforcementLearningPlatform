import datetime
import os
import sys
from matplotlib import pyplot as plt
from numpy import deg2rad
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UavInnerLoop import uav_inner_loop as env
from environment.UavRobust.ref_cmd import generate_uncertainty
from environment.UavRobust.uav import uav_param
from environment.UavRobust.FNTSMC import fntsmc_param
from utils.classes import *

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'PPO-UavInnerLoop'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(2162)

'''Parameter list of the quadrotor'''
DT = 0.01
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
# 只有姿态时范围可以给大点方便训练
uav_param.att_zone = np.atleast_2d(
    [[-deg2rad(90), deg2rad(90)], [-deg2rad(90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_range):
        super(PPOActorCritic, self).__init__()
        self.state_dim = _state_dim
        self.action_dim = _action_dim
        self.action_range = _action_range

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_reset_orthogonal()
        self.critic_reset_orthogonal()
        self.device = 'cpu'
        self.to(self.device)

    def actor_reset_orthogonal(self):
        nn.init.orthogonal_(self.actor[0].weight, gain=1.0)
        nn.init.constant_(self.actor[0].bias, val=1e-3)
        nn.init.orthogonal_(self.actor[2].weight, gain=1.0)
        nn.init.constant_(self.actor[2].bias, val=1e-3)
        nn.init.orthogonal_(self.actor[4].weight, gain=0.01)
        nn.init.constant_(self.actor[4].bias, val=1e-3)

    def critic_reset_orthogonal(self):
        nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
        nn.init.constant_(self.critic[0].bias, val=1e-3)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.constant_(self.critic[2].bias, val=1e-3)
        nn.init.orthogonal_(self.critic[4].weight, gain=1.0)
        nn.init.constant_(self.critic[4].bias, val=1e-3)

    def forward(self):
        raise NotImplementedError

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            action_mean = self.actor(t_state)
        return action_mean.detach()

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.action_dim):
            a = min(max(action[i], -1), 1)
            maxa = self.action_range[i][1]
            mina = self.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return np.array(linear_action)


if __name__ == '__main__':
    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)

    env = env(uav_param, fntsmc_param(),
              ref_amplitude=np.array([np.pi / 3, np.pi / 3, np.pi / 2]),
              ref_period=np.array([4, 4, 4]),
              ref_bias_a=np.array([0, 0, 0]),
              ref_bias_phase=np.array([0., np.pi / 2, np.pi / 3]))

    policy = PPOActorCritic(env.state_dim, env.action_dim, env.action_range)
    policy.load_state_dict(torch.load(optPath + 'actor-critic'))
    test_num = 2
    r = 0
    # video = cv.VideoWriter('../PPO-4-' + env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 200,
    #                        (env.att_w, env.att_h))
    for _ in range(test_num):
        env.reset(random=True)
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            _action_from_actor = policy.evaluate(env.current_state)
            _action = policy.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将actor输出动作转换到实际动作范围
            uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
            env.step_update(_action)  # 环境更新的动作必须是实际物理动作
            r += env.reward
            env.visualization()
            # video.write(env.att_image)
        print(r)
        env.collector.plot_att()
        plt.legend()
        plt.show()
    # video.release()
