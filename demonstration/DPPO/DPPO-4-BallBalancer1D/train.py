import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from BallBalancer1D import BallBalancer1D as env
from algorithm.policy_base.Distributed_PPO import Distributed_PPO as DPPO
from algorithm.policy_base.Distributed_PPO import Worker
from utils.classes import *
import torch.multiprocessing as mp

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'DPPO-BallBalancer1D'


# 每个cpu核上只运行一个进程
os.environ["OMP_NUM_THREADS"] = "1"


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
        super(PPOActorCritic, self).__init__()
        self.checkpoint_file = chkpt_dir + name + '_ppo'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
        self.action_dim = _action_dim
        self.state_dim = _state_dim
        self.action_std_init = _action_std_init
        # 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
        self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.device = 'cpu'
        self.to(self.device)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, s):
        action_mean = self.actor(s)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        _a = dist.sample()
        action_logprob = dist.log_prob(_a)
        state_val = self.critic(s)

        return _a.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, s, a):
        action_mean = self.actor(s)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
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
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                          '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)

    RETRAIN = False  # 基于之前的训练结果重新训练

    env = env()

    '''1. 启动多进程'''
    mp.set_start_method('spawn', force=True)

    '''2. 定义 DPPO 机器基本参数'''
    '''
        这里需要注意，多进程学习的时候学习率要适当下调。主观来讲，如果最开始学习的方向是最优的，那么学习率不减小也没事，控制器肯定收敛特别快，毕竟多进程学习。
        但是如果最开始的方向不好，那么如果不下调学习率，就会导致：
            1. 网络朝着不好的方向走的特别快
            2. 多个进程都学习得很 “快”。local_A 刚朝着 a 的方向走了挺大一步，然后 local_B 又朝着 b 的方向掰了一下。
            这下 global 就懵逼了，没等 global 缓过来呢，local_C，local_D，local_E 咋咋呼呼地就来了，每个人都朝自己的方向走一大步，global 直接崩溃了。
        所以，多进程时，学习率要适当下降，同时需要下调每次学习的网络更新次数 K_epo。理由也同样。走的长度等于每一步长度乘以次数，如果学习率很小，但是每次走一万步，
        global 也得懵逼。
        对于很简单的任务，多进程不见得好。一个人能干完的事，非得10个人干，再加一个监督者，一共11个人，不好管。
        多进程学习适用与那些奖励函数明明给得很合理，但是就是学不出来的环境。实在是没办法了，同时多一些人出去探索探索，集思广益，一起学习。
        但是还是要注意，每个人同时不要走太远，不要走太快，稳稳当当一步一步来。
        脑海中一定要有这么个观念：从完成任务的目的出发，policy-based 算法的多进程、value-based 算法的经验池，都是一种牛逼但是 “无奈” 之举。
    '''
    process_num = 6
    actor_lr = 3e-4 / process_num
    critic_lr = 1e-3 / process_num
    action_std = 0.6
    k_epo_init = 100
    agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=process_num, path=simulationPath)

    '''3. 重新加载全局网络和优化器，这是必须的操作，因为考虑到不同的学习环境要设计不同的网络结构，在训练前，要重写 PPOActorCritic 类'''
    agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'GlobalPolicy',
                                         simulationPath)
    agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'EvalPolicy',
                                       simulationPath)
    agent.global_policy.share_memory()
    agent.optimizer = SharedAdam([
        {'params': agent.global_policy.actor.parameters(), 'lr': actor_lr},
        {'params': agent.global_policy.critic.parameters(), 'lr': critic_lr}
    ])

    '''4. 添加进程'''
    ppo_msg = {'gamma': 0.99, 'k_epo': int(k_epo_init / process_num * 1.5), 'eps_c': 0.2, 'a_std': 0.6,
               'device': 'cpu', 'loss': nn.MSELoss()}
    for i in range(agent.num_of_pro):
        w = Worker(g_pi=agent.global_policy,
                   l_pi=PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'LocalPolicy',
                                       simulationPath),
                   g_opt=agent.optimizer,
                   g_train_n=agent.global_training_num,
                   _index=i,
                   _name='worker' + str(i),
                   _env=env,  # 或者直接写env，随意
                   _queue=agent.queue,
                   _lock=agent.lock,
                   _ppo_msg=ppo_msg)
        agent.add_worker(w)
    agent.DPPO_info()

    '''5. 启动多进程'''
    '''
        五个学习进程，一个评估进程，一共六个。
        学习进程结束会释放标志，当评估进程收集到五个标志时，评估结束。
        评估结束时，评估程序跳出 while True 死循环，整体程序结束。
        结果存储在 simulationPath 中，评估过程中自动存储，不用管。
    '''
    agent.start_multi_process()
