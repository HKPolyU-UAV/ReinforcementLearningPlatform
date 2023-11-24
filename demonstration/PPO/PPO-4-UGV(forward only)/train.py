import datetime
import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UGV import UGV as env
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from utils.classes import *

optPath = './datasave/net'
show_per = 1
timestep = 0
ENV = 'PPO-UGV(forward only)'


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
        self.action_var = torch.Tensor(self.action_std_init ** 2)

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
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))


if __name__ == '__main__':
    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)
    RETRAIN = False

    env = env()
    reward_norm = Normalization(shape=1)

    action_std_init = (env.action_range[:, 1] - env.action_range[:, 0]) / 2 / 3
    '''重新加载Policy网络结构，这是必须的操作'''
    policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init)
    policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init)
    env_msg = {'name': env.name, 'state_dim': env.state_dim, 'action_dim': env.action_dim, 'action_num': env.action_num,
               'action_range': env.action_range}
    agent = PPO(env_msg=env_msg,
                actor_lr=1e-4,
                critic_lr=1e-3,
                gamma=0.99,
                K_epochs=30,
                eps_clip=0.2,
                action_std_init=action_std_init,
                buffer_size=int(env.time_max / env.dt * 4),
                policy=policy,
                policy_old=policy_old,
                path=simulation_path)
    if RETRAIN:
        agent.policy.load_state_dict(torch.load('actor-critic'))
        agent.policy_old.load_state_dict(torch.load('actor-critic'))
        '''如果修改了奖励函数，则原来的critic网络已经不起作用了，需要重新初始化'''
        agent.policy.critic_reset_orthogonal()
        agent.policy_old.critic_reset_orthogonal()
    agent.PPO_info()

    max_training_timestep = int(env.time_max / env.dt) * 40000
    action_std_decay_freq = int(env.time_max / env.dt) * 200
    action_std_decay_rate = 0.05
    min_action_std = 0.1

    sumr = 0
    start_eps = 0
    train_num = 1
    test_num = 0
    test_reward = []
    index = 0
    while timestep <= max_training_timestep:
        env.reset(random=True)
        sumr = 0.
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            action_from_actor, a_log_prob = agent.choose_action(env.current_state)
            action = agent.action_linear_trans(action_from_actor)
            env.step_update(action)  # 环境更新的动作必须是实际物理动作
            sumr += env.reward
            '''存数'''
            agent.buffer.append(s=env.current_state,
                                a=action_from_actor,
                                log_prob=a_log_prob,
                                r=reward_norm(env.reward),
                                s_=env.next_state,
                                done=1.0 if env.is_terminal else 0.0,
                                success=1.0 if env.terminal_flag == 1 else 0.0,
                                index=index)
            index += 1
            timestep += 1
            '''学习'''
            if timestep % agent.buffer.batch_size == 0:
                print('========= Training =========')
                print('Episode: {}'.format(agent.episode))
                print('Num of learning: {}'.format(train_num))
                agent.learn()
                train_num += 1
                start_eps = agent.episode
                index = 0
                if train_num % 20 == 0 and train_num > 0:
                    print('========= Testing =========')
                    average_test_r = 0
                    test_num += 1
                    for _ in range(3):
                        env.reset(random=True)
                        while not env.is_terminal:
                            env.current_state = env.next_state.copy()
                            action_from_actor, a_log_prob = agent.choose_action(env.current_state)
                            action = agent.action_linear_trans(action_from_actor)
                            env.step_update(action)  # 环境更新的动作必须是实际物理动作
                            average_test_r += env.reward
                            env.visualization()
                    average_test_r /= 3
                    test_reward.append(average_test_r)
                    print('   Evaluating %.0f | Reward: %.2f ' % (test_num, average_test_r))
                    temp = simulation_path + 'test_num' + '_' + str(test_num - 1) + '_save/'
                    os.mkdir(temp)
                    pd.DataFrame({'reward': test_reward}).to_csv(simulation_path + 'retrain_reward.csv')
                    time.sleep(0.01)
                    agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
            if timestep % action_std_decay_freq == 0:
                ratio = max(1 - timestep / action_std_decay_freq * action_std_decay_rate, min_action_std)
                agent.set_action_std(ratio * action_std_init)
        if agent.episode % 5 == 0:
            print('Episode: ', agent.episode, ' Reward: ', sumr)
        agent.episode += 1
