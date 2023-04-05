import os
import sys
import datetime
import cv2 as cv
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from environment.envs.UGV.ugv_forward_continuous import UGV_Forward_Continuous as env
from algorithm.policy_base.Distributed_PPO import Distributed_PPO as DPPO
from algorithm.policy_base.Distributed_PPO import Worker
from common.common_cls import *
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ENV = 'DPPO-UGVForwardContinuous'


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(3407)
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
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
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
    log_dir = '../../../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                          '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    env = env(initPhi=0., save_cfg=False, x_size=5, y_size=5, start=[1,1], terminal=[4,4])

    if TRAIN:
        '''1. 启动多进程'''
        mp.set_start_method('spawn', force=True)

        '''2. 定义 DPPO 及其基本参数'''
        process_num = 10
        actor_lr = 3e-4 / min(process_num, 5)
        critic_lr = 1e-3 / min(process_num, 5)  # 一直都是 1e-3
        action_std = 0.9
        k_epo = int(100 / process_num * 2)  # int(100 / process_num * 1.1)
        agent = DPPO(env=env, actor_lr=actor_lr, critic_lr=critic_lr, num_of_pro=process_num, path=simulationPath)

        '''3. 重新加载全局网络和优化器，这是必须的操作，因为考虑到不同的学习环境要设计不同的网络结构，在训练前，要重写 PPOActorCritic 类'''
        agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'GlobalPolicy',
                                             simulationPath)
        agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'EvalPolicy',
                                           simulationPath)
        if RETRAIN:
            agent.global_policy.load_state_dict(torch.load('Policy_PPO-2-88800'))
        agent.global_policy.share_memory()
        agent.optimizer = SharedAdam([
            {'params': agent.global_policy.actor.parameters(), 'lr': actor_lr},
            {'params': agent.global_policy.critic.parameters(), 'lr': critic_lr}
        ])

        '''4. 添加进程'''
        ppo_msg = {'gamma': 0.99, 'k_epo': k_epo, 'eps_c': 0.2, 'a_std': action_std, 'device': 'cpu',
                   'loss': nn.MSELoss()}
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
    else:
        agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulationPath)
        agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1, 'GlobalPolicy',
                                             simulationPath)
        agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1, 'EvalPolicy', simulationPath)
        # agent.load_models(optPath + 'DPPO-4-SecondOrderIntegration/')
        agent.global_policy.load_state_dict(torch.load('Policy_PPO92700'))
        agent.eval_policy.load_state_dict(agent.global_policy.state_dict())
        test_num = 100
        error = []
        terminal_list = []
        # cap = cv.VideoWriter('record.mp4', cv.VideoWriter_fourcc(*'mp4v'), 120, (env.image_size[0]-env.board, env.image_size[1]))
        for i in range(test_num):
            env.reset_random()
            # env.reset()
            if i % 100 == 0:
                print(i)
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                action_from_actor = agent.evaluate(env.current_state)
                action_from_actor = action_from_actor.numpy()
                action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                env.show_dynamic_image(isWait=False)  # 画图
            # cap.write(env.image[:, 0:env.image_size[0] - env.board])
            # error.append(np.linalg.norm(env.error))
            # terminal_list.append(env.init_target)
        # cap.release()
        '''统计一下，没有什么特殊的'''
    # error = np.array(error)
    # terminal_list = np.array(terminal_list)
    # norm_error = (error - np.min(error)) / (np.max(error) - np.min(error))
    # color = []
    # for _e in norm_error:
    # 	color.append((_e, 0., 0.))
    # print('Mean error  ', error.mean())
    # print('Std error   ', error.std())
    # print('Max error   ', np.max(error))
    # print('Min error   ', np.min(error))
    # plt.figure(0)
    # plt.plot(range(test_num), error)
    # plt.ylim(0, 0.35)
    # plt.yticks(np.arange(0, 0.35, 0.05))
    #
    # plt.figure(1)
    # plt.hist(error, rwidth=0.05)
    #
    # print('分布图')
    # plt.figure(2)
    # plt.scatter(x=terminal_list[:, 0], y=terminal_list[:, 1], marker='o', c=color, s=[25 for _ in range(test_num)])		#
    # plt.axis('equal')
    # plt.xlim(0, env.map_size[0])
    # plt.ylim(0, env.map_size[1])
    # plt.xticks(np.arange(0, 1, env.map_size[0]))
    # plt.yticks(np.arange(0, 1, env.map_size[1]))
    # plt.show()
