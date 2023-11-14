import torch
from utils.functions import *
from utils.classes import Actor, Critic, ReplayBuffer, GaussianNoise
import torch.nn.functional as func

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Twin_Delayed_DDPG:
    def __init__(self,
                 env,
                 gamma: float = 0.9,
                 noise_clip: float = 1 / 2,
                 noise_policy: float = 1 / 4,
                 policy_delay: int = 5,
                 critic1_soft_update: float = 1e-2,
                 critic2_soft_update: float = 1e-2,
                 actor_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 actor: Actor = Actor(),
                 target_actor: Actor = Actor(),
                 critic1: Critic = Critic(),
                 target_critic1: Critic = Critic(),
                 critic2: Critic = Critic(),
                 target_critic2: Critic = Critic(),
                 path: str = ''):
        self.env = env
        '''Twin-Delay-DDPG'''
        self.gamma = gamma
        # for target policy smoothing regularization
        self.noise_clip = noise_clip
        self.noise_policy = noise_policy
        self.action_regularization = GaussianNoise(mu=np.zeros(self.env.action_dim))
        # for target policy smoothing regularization
        self.policy_delay = policy_delay
        self.policy_delay_iter = 0
        self.critic1_tau = critic1_soft_update
        self.critic2_tau = critic2_soft_update
        self.actor_tau = actor_soft_update
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.env.state_dim, self.env.action_dim)
        self.path = path
        '''Twin-Delay-DDPG'''

        '''network'''
        self.actor = actor
        self.target_actor = target_actor

        self.critic1 = critic1
        self.target_critic1 = target_critic1

        self.critic2 = critic2
        self.target_critic2 = target_critic2
        self.actor_replace_iter = 0
        '''network'''

        self.noise_gaussian = GaussianNoise(mu=np.zeros(self.env.action_dim))
        self.update_network_parameters()

        self.episode = 0
        self.reward = 0

        self.save_episode = []  # 保存的每一个回合的回合数
        self.save_reward = []  # 保存的每一个回合的奖励
        self.save_time = []
        self.save_average_reward = []  # 保存的每一个回合的平均时间的奖励
        self.save_successful_rate = []
        self.save_step = []  # 保存的每一步的步数
        self.save_stepreward = []  # 保存的每一步的奖励

    def choose_action_random(self):
        """
        :brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
        :return:    random action
        """
        return np.random.uniform(low=-1, high=1, size=self.env.action_dim)

    def choose_action(self, state, is_optimal=False, sigma=1 / 3):
        self.actor.eval()  # 切换到测试模式
        t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
        mu = self.actor(t_state).to(self.actor.device)  # choose action
        if is_optimal:
            mu_prime = mu
        else:
            mu_prime = mu + torch.tensor(self.noise_gaussian(sigma=sigma), dtype=torch.float).to(
                self.actor.device)  # action with gaussian noise
            # mu_prime = mu + torch.tensor(self.noise_OU(), dtype=torch.float).to(self.actor.device)             # action with OU noise
        self.actor.train()  # 切换回训练模式
        mu_prime_np = mu_prime.cpu().detach().numpy()
        return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

    def evaluate(self, state):
        self.target_actor.eval()
        t_state = torch.tensor(state, dtype=torch.float).to(self.target_actor.device)  # get the tensor of the state
        act = self.target_actor(t_state).to(self.target_actor.device)  # choose action
        return act.cpu().detach().numpy()

    def learn(self, is_reward_ascent=True, critic_random=True):
        if self.memory.mem_counter < self.memory.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        state = torch.tensor(state, dtype=torch.float).to(self.critic1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic1.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic1.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic1.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic1.device)

        '''这里TD3的不同Critic网络，默认在同一块GPU上，当然......我就有一块GPU'''
        # state = torch.tensor(state, dtype=torch.float).to(self.critic2.device)
        # action = torch.tensor(action, dtype=torch.float).to(self.critic2.device)
        # reward = torch.tensor(reward, dtype=torch.float).to(self.critic2.device)
        # new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic2.device)
        # done = torch.tensor(done, dtype=torch.float).to(self.critic2.device)
        '''这里TD3的不同Critic网络，默认在同一块GPU上，当然......我就有一块GPU'''

        self.target_actor.eval()  # PI'
        self.target_critic1.eval()  # Q1'
        self.critic1.eval()  # Q1
        self.target_critic2.eval()  # Q2'
        self.critic2.eval()  # Q2

        target_actions = self.target_actor.forward(new_state).to(self.critic1.device)  # a' = PI'(s')
        '''动作正则化'''
        action_noise = torch.clip(torch.tensor(self.action_regularization(sigma=self.noise_policy)), -self.noise_clip,
                                  self.noise_clip).to(self.critic1.device)
        target_actions += action_noise
        '''动作正则化'''
        critic_value1_ = self.target_critic1.forward(new_state, target_actions)
        critic_value1 = self.critic1.forward(state, action)
        critic_value2_ = self.target_critic2.forward(new_state, target_actions)
        critic_value2 = self.critic2.forward(state, action)

        '''
        Attention please!
        这里的target变量最开始的实现是用list的方式实现，具体如下：
        target = []
        for j in range(self.memory.batch_size):
            target.append(reward[j] + self.gamma * torch.minimum(critic_value1_[j], critic_value2_[j]) * done[j])
        如此实现，使得learn函数中将近90%的时间被这个循环所占用。因此，将target这个变量直接用tensor的方式去构建，具体如下：
        target = reward + self.gamma * torch.minimum(critic_value1_.squeeze(), critic_value2_.squeeze()) * done
        为防止搞错tensor的维度，将记录的维度列在下边
                reward:           torch.Size([batch])
            critic_value1_:       torch.Size([batch, 1])
            critic_value2_:       torch.Size([batch, 1])
                done:             torch.Size([batch])
               target:            torch.Size([batch])
          .view之前的target:    torch.Size([batch])
          .view之后的target:    torch.Size([batch, 1])
        经验：数据处理，千万不要使用list，用numpy或者tensor都行。
        '''
        '''取较小的Q'''
        target = torch.tensor(
            reward + self.gamma * torch.minimum(critic_value1_.squeeze(), critic_value2_.squeeze()) * done).to(
            self.critic1.device)

        target1 = target.view(self.memory.batch_size, 1)
        target2 = target.view(self.memory.batch_size, 1)

        '''critic1 training'''
        self.critic1.train()
        self.critic1.optimizer.zero_grad()
        critic_loss = func.mse_loss(target1, critic_value1)
        critic_loss.backward()
        self.critic1.optimizer.step()
        '''critic1 training'''

        '''critic2 training'''
        self.critic2.train()
        self.critic2.optimizer.zero_grad()
        critic_loss = func.mse_loss(target2, critic_value2)
        critic_loss.backward()
        self.critic2.optimizer.step()
        '''critic2 training'''

        self.policy_delay_iter += 1

        '''actor training, choose critic1 or critic2 randomly'''
        '''延迟更新'''
        if self.policy_delay_iter % self.policy_delay == 0:
            if critic_random:
                if np.random.randint(1, 2) == 1:
                    self.critic1.eval()
                    self.actor.optimizer.zero_grad()
                    mu = self.actor.forward(state)
                    self.actor.train()
                    actor_loss = -self.critic1.forward(state, mu)
                else:
                    self.critic2.eval()
                    self.actor.optimizer.zero_grad()
                    mu = self.actor.forward(state)
                    self.actor.train()
                    actor_loss = -self.critic2.forward(state, mu)
            else:
                self.critic1.eval()
                self.actor.optimizer.zero_grad()
                mu = self.actor.forward(state)
                self.actor.train()
                actor_loss = -self.critic1.forward(state, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()
        '''actor training, choose critic1 or critic2 randomly'''

        self.update_network_parameters()

    def update_network_parameters(self, is_target_critics_delay: bool = False):
        """
        :return:        None
        """
        if not is_target_critics_delay:
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.critic1_tau) + param.data * self.critic1_tau)  # soft update
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.critic2_tau) + param.data * self.critic2_tau)  # soft update

        if self.policy_delay_iter % self.policy_delay == 0:
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update
            if is_target_critics_delay:
                for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.critic1_tau) + param.data * self.critic1_tau)  # soft update
                for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.critic2_tau) + param.data * self.critic2_tau)

    def save_ac(self, msg, path):
        torch.save(self.actor.state_dict(), path + 'actor' + msg)
        torch.save(self.target_actor.state_dict(), path + 'target_actor' + msg)
        torch.save(self.critic1.state_dict(), path + 'critic1' + msg)
        torch.save(self.target_critic1.state_dict(), path + 'target_critic1' + msg)
        torch.save(self.critic2.state_dict(), path + 'critic2' + msg)
        torch.save(self.target_critic2.state_dict(), path + 'target_critic2' + msg)

    def TD3_info(self):
        print('agent name：', self.env.name)
        print('state_dim:', self.env.state_dim)
        print('action_dim:', self.env.action_dim)
        print('action_range:', self.env.action_range)

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.env.action_dim):
            a = min(max(action[i], -1), 1)
            maxa = self.env.action_range[i][1]
            mina = self.env.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return linear_action
