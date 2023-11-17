import datetime
import os
import sys
from numpy import deg2rad

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UavInnerLoop import uav_inner_loop as env
from environment.UavRobust.uav import uav_param
from environment.UavRobust.FNTSMC import fntsmc_param
from Distributed_PPO import Distributed_PPO as DPPO
from Distributed_PPO import Worker
from utils.classes import *
import torch.multiprocessing as mp

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'DPPO-UavInnerLoop'

# 每个cpu核上只运行一个进程
os.environ["OMP_NUM_THREADS"] = "1"

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
uav_param.att_zone = np.atleast_2d(
        [[-deg2rad(90), deg2rad(90)], [-deg2rad(90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_std_init):
        super(PPOActorCritic, self).__init__()
        self.state_dim = _state_dim
        self.action_dim = _action_dim
        self.action_std_init = _action_std_init
        self.action_var = torch.Tensor(self.action_std_init ** 2)

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
        self.action_var = torch.Tensor(new_action_std ** 2)

    def forward(self):
        raise NotImplementedError

    def act(self, s):
        """选取动作"""
        action_mean = self.actor(s)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        _a = dist.sample()
        action_logprob = dist.log_prob(_a)

        return _a.detach(), action_logprob.detach()

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
        torch.save(self.state_dict(), path + name + str(num))


if __name__ == '__main__':
    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)

    RETRAIN = False

    ref_amplitude = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
    ref_period = np.array([4, 4, 4])
    ref_bias_a = np.array([0, 0, 0])
    ref_bias_phase = np.array([0., np.pi / 2, np.pi / 3])
    env = env(uav_param, fntsmc_param(), ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    env.msg_print_flag = False  # 别疯狂打印出界了

    '''1. 启动多进程'''
    mp.set_start_method('spawn', force=True)
    '''2. 定义DPPO参数'''
    process_num = 10
    actor_lr = 1e-5 / min(process_num, 5)
    critic_lr = 1e-4 / min(process_num, 5)
    action_std_init = (env.action_range[:, 1] - env.action_range[:, 0]) / 2 / 3
    k_epo_init = 100
    agent = DPPO(env=env, actor_lr=actor_lr, critic_lr=critic_lr, num_of_pro=process_num, path=simulation_path)
    '''3. 重新加载全局网络和优化器，这是必须的操作，考虑到不同学习环境设计不同的网络结构，训练前要重写PPOActorCritic'''
    agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std_init)
    agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std_init)
    if RETRAIN:
        agent.global_policy.load_state_dict(torch.load('Policy_PPO47400'))
        '''如果修改了奖励函数，则原来的critic网络已经不起作用了，需要重新初始化'''
        agent.global_policy.critic_reset_orthogonal()
    agent.global_policy.share_memory()
    agent.optimizer = SharedAdam([
        {'params': agent.global_policy.actor.parameters(), 'lr': actor_lr},
        {'params': agent.global_policy.critic.parameters(), 'lr': critic_lr}
    ])
    '''4. 添加进程'''
    ppo_msg = {'gamma': 0.99, 'k_epo': int(k_epo_init / process_num * 1.5), 'eps_c': 0.2, 'a_std': action_std_init,
               'device': 'cpu', 'loss': nn.MSELoss()}
    for i in range(agent.num_of_pro):
        worker = Worker(g_pi=agent.global_policy,
                        l_pi=PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std_init),
                        g_opt=agent.optimizer,
                        g_train_n=agent.global_training_num,
                        _index=i,
                        _name='worker' + str(i),
                        _env=env,
                        _queue=agent.queue,
                        _lock=agent.lock,
                        _ppo_msg=ppo_msg)
        agent.add_worker(worker)
    agent.DPPO_info()
    '''5. 原神(多进程学习)启动'''
    """
    多个学习进程和一个评估进程，学习进程在结束后会释放标志，评估进程收集到所有学习进程标志时结束评估，
    评估结束时，评估程序会跳出while死循环，整个程序结束，结果在评估过程中自动存储在simulation_path中。
    """
    agent.start_multi_process()
