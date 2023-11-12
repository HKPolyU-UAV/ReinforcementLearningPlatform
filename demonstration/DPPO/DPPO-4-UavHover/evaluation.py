import datetime
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from UavHover import uav_hover as env
from environment.uav_robust.uav import uav_param
from environment.uav_robust.FNTSMC import fntsmc_param
from Distributed_PPO import Distributed_PPO as DPPO
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
    def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
        super(PPOActorCritic, self).__init__()
        self.checkpoint_file = chkpt_dir + name + '_ppo'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
        self.state_dim = _state_dim
        self.action_dim = _action_dim
        self.action_std_init = _action_std_init
        self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)

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
            nn.Linear(64, 1),
        )
        self.device = 'cpu'
        self.to(self.device)

    def set_action_std(self, new_action_std):
        """手动设置动作方差"""
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, s):
        """选取动作"""
        action_mean = self.actor(s)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        _a = dist.sample()
        action_logprob = dist.log_prob(_a)
        state_val = self.critic(s)

        return _a.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, s, a):
        """评估状态动作价值"""
        action_mean = self.actor(s)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # 一维动作单独处理
        if self.action_dim == 1:
            a = a.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(a)
        dist_entropy = dist.entropy()
        state_values = self.critic(s)

        return action_logprobs, state_values, dist_entropy

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
    pos_agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulation_path)
    pos_agent.global_policy = PPOActorCritic(int(pos_agent.env.state_dim / 2), int(pos_agent.env.action_dim / 2), 0.1,    # 这里环境维数是内外环一起的维数，需要减半
                                             'GlobalPolicy_ppo', simulation_path)
    pos_agent.eval_policy = PPOActorCritic(int(pos_agent.env.state_dim / 2), int(pos_agent.env.action_dim / 2), 0.1,
                                           'EvalPolicy_ppo', simulation_path)
    pos_agent.load_models(optPath + 'pos-actor-critic')
    pos_agent.eval_policy.load_state_dict(pos_agent.global_policy.state_dict())
    '''2. 内环控制器加载'''
    att_agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulation_path)
    att_agent.global_policy = PPOActorCritic(int(att_agent.env.state_dim / 2), int(att_agent.env.action_dim / 2), 0.1,
                                             'GlobalPolicy_ppo', simulation_path)
    att_agent.eval_policy = PPOActorCritic(int(att_agent.env.state_dim / 2), int(att_agent.env.action_dim / 2), 0.1,
                                           'EvalPolicy_ppo', simulation_path)
    att_agent.load_models(optPath + 'att-actor-critic')
    att_agent.eval_policy.load_state_dict(att_agent.global_policy.state_dict())
    '''3. 开始测试'''
    env.msg_print_flag = True
    test_num = 5
    for _ in range(test_num):
        env.reset_random()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            '''3.1 外环网络根据外环状态给出虚拟加速度'''
            u_from_actor = pos_agent.evaluate(env.current_state[:6]).numpy()
            '''3.2 内环网络根据内环状态给出转矩'''
            torque_from_actor = att_agent.evaluate(env.current_state[6:]).numpy()
            '''3.3 拼接成6维action，传入环境'''
            action = np.concatenate((u_from_actor, torque_from_actor))
            action = pos_agent.action_linear_trans(action.flatten())    # 两个agent用的同一个环境，动作变换是统一的，用哪个都一样
            env.step_update(action)  # 环境更新的动作必须是实际物理动作
            env.visualization()
        env.collector.plot_pos()
        env.collector.plot_vel()
        env.collector.plot_throttle()
        env.collector.plot_att()
        env.collector.plot_pqr()
        env.collector.plot_torque()
        plt.show()
