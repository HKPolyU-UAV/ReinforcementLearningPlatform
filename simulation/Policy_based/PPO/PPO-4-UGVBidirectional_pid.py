import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from environment.envs.UGV_PID.UGVBidirectional_pid import UGV_Bidirectional_PID
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ALGORITHM = 'PPO'
ENV = 'UGVBidirectional_pid'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(3407)


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
        """
		@param _state_dim:
		@param _action_dim:
		@param _action_std_init:
		@param name:
		@param chkpt_dir:
		"""
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
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        # torch.cuda.empty_cache()
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
                                                          '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    env = UGV_Bidirectional_PID(pos0=np.array([1.0, 1.0]),
                                vel0=np.array([0.0, 0.0]),
                                phi0=0.,
                                omega0=0.,
                                map_size=np.array([5.0, 5.0]),
                                target=np.array([2.5, 2.5]))

    if TRAIN:
        action_std_init = 0.8
        '''重新加载Policy网络结构，这是必须的操作'''
        policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
        policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
        agent = PPO(env=env,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    gamma=0.99,
                    K_epochs=10,
                    eps_clip=0.2,
                    action_std_init=action_std_init,
                    buffer_size=int(env.timeMax / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
                    policy=policy,
                    policy_old=policy_old,
                    path=simulationPath)
        '''重新加载Policy网络结构，这是必须的操作'''

        agent.PPO_info()

        max_training_timestep = int(env.timeMax / env.dt) * 10000  # 10000回合
        action_std_decay_freq = int(5e6)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

        sumr = 0
        start_eps = 0
        train_num = 0
        test_num = 0
        index = 0
        while timestep <= max_training_timestep:
            env.reset_random()
            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                # print(env.current_state)
                action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
                action_from_actor = action_from_actor.numpy()
                action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
                env.step_update(action)  # 环境更新的action需要是物理的action
                # env.show_dynamic_image(isWait=False)  # 画图
                sumr += env.reward
                '''存数'''
                agent.buffer.append(s=env.current_state,
                                    a=action_from_actor,
                                    log_prob=a_log_prob.numpy(),
                                    r=env.reward,
                                    sv=s_value.numpy(),
                                    done=1.0 if env.is_terminal else 0.0,
                                    index=index)
                index += 1
                timestep += 1
                '''存数'''
                '''学习'''
                if timestep % agent.buffer.batch_size == 0:
                    print('========== LEARN ==========')
                    print('Episode: {}'.format(agent.episode))
                    print('Num of learning: {}'.format(train_num))
                    agent.learn()
                    '''clear buffer'''
                    # agent.buffer.clear()
                    average_train_r = round(sumr / (agent.episode + 1 - start_eps), 3)
                    print('Average reward:', average_train_r)
                    # agent.writer.add_scalar('train_r', average_train_r, train_num)		# to tensorboard
                    train_num += 1
                    start_eps = agent.episode
                    sumr = 0
                    index = 0
                    if train_num % 100 == 0 and train_num > 0:
                        average_test_r = agent.agent_evaluate(5)
                        # agent.writer.add_scalar('test_r', average_test_r, test_num)	# to tensorboard
                        test_num += 1
                        print('check point save')
                        temp = simulationPath + 'episode' + '_' + str(agent.episode) + '_save/'
                        os.mkdir(temp)
                        time.sleep(0.01)
                        agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
                    print('========== LEARN ==========')
                '''学习'''

                if timestep % action_std_decay_freq == 0:
                    agent.decay_action_std(action_std_decay_rate, min_action_std)
            agent.episode += 1

    if TEST:
        action_std_init = 0.8
        policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
        policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
        agent = PPO(env=env,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    gamma=0.99,
                    K_epochs=10,
                    eps_clip=0.2,
                    action_std_init=action_std_init,
                    buffer_size=int(env.timeMax / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
                    policy=policy,
                    policy_old=policy_old,
                    path=simulationPath)
        agent.policy.load_state_dict(torch.load('../../../datasave/network/PPO-UGV-Bidirectional_pid/parameters'
                                                '/Policy_PPO720000'))
        r = agent.agent_evaluate(test_num=10)
