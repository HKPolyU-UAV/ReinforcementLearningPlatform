import datetime
import os
import sys
from matplotlib import pyplot as plt
from numpy import deg2rad

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from env import uav_inner_loop as env
from environment.uav_robust.ref_cmd import generate_uncertainty
from environment.uav_robust.uav import uav_param
from environment.uav_robust.FNTSMC import fntsmc_param
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from utils.classes import *

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'PPO-uav-inner-loop'


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
        cov_mat = torch.diag_embed(self.action_var).to(self.device)
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

    action_std_init = 0.8
    policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulation_path)
    policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulation_path)
    agent = PPO(env=env,
                actor_lr=3e-4,
                critic_lr=1e-3,
                gamma=0.99,
                K_epochs=20,
                eps_clip=0.2,
                action_std_init=action_std_init,
                buffer_size=int(env.time_max / env.dt * 2),
                policy=policy,
                policy_old=policy_old,
                path=simulation_path)
    agent.policy.load_state_dict(torch.load(optPath + 'actor-critic'))
    test_num = 1
    r = 0
    ux, uy, uz = [], [], []
    for _ in range(test_num):
        # env.reset()
        env.reset_random()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            _action_from_actor = agent.evaluate(env.current_state)
            _action = agent.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将actor输出动作转换到实际动作范围
            uncertainty = generate_uncertainty(time=env.time, is_ideal=True)  # 生成干扰信号
            env.step_update(_action)  # 环境更新的动作必须是实际物理动作
            r += env.reward
            env.visualization()
            ux.append(_action[0])
            uy.append(_action[1])
            uz.append(_action[2])
            # print(_action)
            # env.uav_vis.render(uav_pos=env.uav_pos(),
            #                    uav_pos_ref=env.pos_ref,
            #                    uav_att=env.uav_att(),
            #                    uav_att_ref=env.att_ref,
            #                    d=4 * env.d)  # to make it clearer, we increase the size 4 times
            # rate.sleep()
        print(r)
        env.collector.plot_att()
        plt.plot(ux, label='ux')
        plt.plot(uy, label='uy')
        plt.plot(uz, label='uz')
        plt.legend()
        plt.show()
