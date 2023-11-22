import datetime
import os
import sys
import matplotlib.pyplot as plt
import cv2 as cv
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UavHover import uav_hover as env
from environment.UavRobust.uav import uav_param
from environment.UavRobust.FNTSMC import fntsmc_param
from utils.classes import *

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'DPPO-UavHover'


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
uav_param.pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 5]])
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
    # rospy.init_node(name='PPO_uav_hover_outer_loop', anonymous=False)

    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)

    env = env(uav_param, fntsmc_param(), fntsmc_param(), target0=np.array([-1, 3, 2]))

    '''1. 外环控制器加载'''
    pos_policy = PPOActorCritic(int(env.state_dim / 2), int(env.action_dim / 2), env.action_range[:3, :])
    pos_policy.load_state_dict(torch.load(optPath + 'pos-actor-critic'))
    '''2. 内环控制器加载'''
    att_policy = PPOActorCritic(int(env.state_dim / 2), int(env.action_dim / 2), env.action_range[3:, :])
    att_policy.load_state_dict(torch.load(optPath + 'att-actor-critic'))
    '''3. 开始测试'''
    env.msg_print_flag = True
    test_num = 2
    # video = cv.VideoWriter('../DPPO-4-' + env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 200,
    #                        (env.width, env.height))
    for _ in range(test_num):
        env.reset(random=True)
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            '''3.1 外环网络根据外环状态给出虚拟加速度'''
            u_from_actor = pos_policy.evaluate(env.current_state[:6]).numpy()
            uf = pos_policy.action_linear_trans(u_from_actor.flatten())
            '''3.2 内环网络根据内环状态给出转矩'''
            torque_from_actor = att_policy.evaluate(env.current_state[6:]).numpy()
            torque = att_policy.action_linear_trans(torque_from_actor.flatten())
            '''3.3 拼接成6维action，传入环境'''
            action = np.concatenate((uf, torque))
            env.step_update(action)  # 环境更新的动作必须是实际物理动作
            env.visualization()
            # video.write(env.image[:, :env.width])
        env.collector.plot_pos()
        env.collector.plot_vel()
        env.collector.plot_throttle()
        env.collector.plot_att()
        env.collector.plot_pqr()
        env.collector.plot_torque()
        plt.show()
    # video.release()
