import torch
from utils.classes import SACActor, SACCritic, ReplayBuffer
import torch.nn.functional as func

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class SAC:
    def __init__(self,
                 env_msg: dict,
                 gamma: float = 0.99,
                 critic_tau: float = 0.005,
                 memory_capacity: int = 5000,
                 batch_size: int = 256,
                 actor: SACActor = SACActor(),
                 critic: SACCritic = SACCritic(),
                 target_critic: SACCritic = SACCritic(),
                 a_lr: float = 3e-4,
                 c_lr: float = 1e-4,
                 alpha_lr: float = 3e-4,
                 adaptive_alpha: bool = True):
        """SAC"""
        self.env_msg = env_msg
        self.gamma = gamma  # discount factor
        self.tau = critic_tau  # Softly update the target network
        self.memory = ReplayBuffer(memory_capacity, batch_size, env_msg['state_dim'], env_msg['action_dim'])
        """SAC"""

        '''network'''
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.a_lr = a_lr  # actor learning rate
        self.c_lr = c_lr  # critic learning rate
        self.alpha_lr = alpha_lr  # alpha learning rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.device = self.target_critic.device
        '''network'''

        self.adaptive_alpha = adaptive_alpha  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -env_msg['action_dim']
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = 0.2
        self.a_min = torch.FloatTensor(env_msg['action_range'][:, 0])
        self.a_max = torch.FloatTensor(env_msg['action_range'][:, 1])
        self.episode = 0

    def choose_action(self, s, deterministic=False):
        s_t = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _ = self.actor(s_t, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        a = torch.maximum(torch.minimum(a.cpu(), self.a_max), self.a_min)
        return a.data.numpy().flatten()

    def choose_action_random(self):
        a = torch.rand(self.env_msg['action_dim']) * (self.a_max - self.a_min) + self.a_min
        return a.cpu().detach().numpy().flatten()

    def learn(self, is_reward_ascent=False, iter=1):
        for _ in range(iter):
            batch_s, batch_a, batch_r, batch_s_, batch_dw = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
            batch_s = torch.FloatTensor(batch_s).to(self.device)
            batch_a = torch.FloatTensor(batch_a).to(self.device)
            batch_r = torch.FloatTensor(batch_r).unsqueeze(1).to(self.device)
            batch_s_ = torch.FloatTensor(batch_s_).to(self.device)
            batch_dw = torch.FloatTensor(batch_dw).unsqueeze(1).to(self.device)

            with torch.no_grad():
                batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
                # Compute target Q
                target_Q1, target_Q2 = self.target_critic(batch_s_, batch_a_)
                target_Q = batch_r + self.gamma * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

            # Compute current Q
            current_Q1, current_Q2 = self.critic(batch_s, batch_a)
            # Compute critic loss
            critic_loss = func.mse_loss(current_Q1, target_Q) + func.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute actor loss
            a, log_pi = self.actor(batch_s)
            Q1, Q2 = self.critic(batch_s, a)
            Q = torch.min(Q1, Q2)
            actor_loss = (self.alpha * log_pi - Q).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Update alpha
            if self.adaptive_alpha:
                # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

            # Softly update target networks
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_ac(self, msg, path):
        torch.save(self.actor.state_dict(), path + 'actor' + msg)
        torch.save(self.critic.state_dict(), path + 'critic' + msg)
        torch.save(self.target_critic.state_dict(), path + 'target_critic' + msg)

    def SAC_info(self):
        print('agent name：', self.env_msg['name'])
        print('state_dim:', self.env_msg['state_dim'])
        print('action_dim:', self.env_msg['action_dim'])
        print('action_range:', self.env_msg['action_range'])
