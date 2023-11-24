import numpy as np
import torch
import torch.nn as nn

from utils.classes import PPOActorCritic, RolloutBuffer, RolloutBuffer2

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Proximal_Policy_Optimization:
    def __init__(self,
                 env_msg,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 K_epochs: int = 10,
                 eps_clip: float = 0.2,
                 action_std_init: np.ndarray = None,
                 buffer_size: int = 1200,
                 policy: PPOActorCritic = PPOActorCritic(1, 1, np.zeros(3)),
                 policy_old: PPOActorCritic = PPOActorCritic(1, 1, np.zeros(3)),
                 path: str = ''):
        """
		@note:
		@param env:					RL environment
		@param actor_lr:			actor learning rate
		@param critic_lr:			critic learning rate
		@param gamma:				discount factor
		@param K_epochs:			update policy for K epochs in one PPO update
		@param eps_clip:			clip parameter for PPO
		@param action_std_init:		starting std for action distribution (Multivariate Normal)
		@param path:				path
		"""
        self.env_msg = env_msg
        '''PPO'''
        self.gamma = gamma  # discount factor
        self.K_epochs = K_epochs  # 每隔 timestep_num 学习一次
        self.eps_clip = eps_clip
        self.action_std = action_std_init
        self.path = path
        self.buffer = RolloutBuffer(buffer_size, self.env_msg['state_dim'], self.env_msg['action_dim'])
        self.buffer2 = RolloutBuffer2(self.env_msg['state_dim'], self.env_msg['action_dim'])
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        '''PPO'''

        '''networks'''
        # self.policy = PPOActorCritic(self.env.state_dim, self.env.action_dim, action_std_init, name='PPOActorCritic', chkpt_dir=self.path)
        # self.policy_old = PPOActorCritic(self.env.state_dim, self.env.action_dim, action_std_init, name='PPOActorCritic_old', chkpt_dir=self.path)
        self.policy = policy
        self.policy_old = policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': self.critic_lr}
        ])
        self.loss = nn.MSELoss()
        self.device = device  # 建议使用 CPU 训练
        '''networks'''

        self.episode = 0
        self.reward = 0

    # self.writer = SummaryWriter(path)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        print("setting actor output action_std to : ", self.action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def choose_action_random(self):
        """
		:brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
		:return:    random action
		"""
        return np.random.uniform(low=-1, high=1, size=self.env.action_dim)

    def choose_action(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(device)
            action, action_log_prob = self.policy_old.act(t_state)

        return action.detach().cpu().numpy().flatten(), action_log_prob.detach().cpu().numpy().flatten(),

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            action_mean = self.policy.actor(t_state)
        return action_mean.detach()

    def learn(self):
        """
		@note: 	 network update
		@return: None
		"""
        '''1. Monte Carlo estimate of returns'''
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.r), reversed(self.buffer.done)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        '''2. Normalizing the rewards'''
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        '''3. convert numpy to tensor'''
        with torch.no_grad():
            old_states = torch.FloatTensor(self.buffer.s).detach().to(self.device)
            old_actions = torch.FloatTensor(self.buffer.a).detach().to(self.device)
            old_log_probs = torch.FloatTensor(self.buffer.a_lp).detach().to(self.device)
            old_state_values = torch.FloatTensor(self.policy.critic(torch.FloatTensor(self.buffer.s))).detach().to(self.device)

        '''4. calculate advantages'''
        advantages = torch.squeeze(rewards).detach() - torch.squeeze(old_state_values).detach()

        '''5. Optimize policy for K epochs'''
        for _ in range(self.K_epochs):
            '''5.1 Evaluating old actions and values'''
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            '''5.2 match state_values tensor dimensions with rewards tensor'''
            state_values = torch.squeeze(state_values)
            rewards = torch.squeeze(rewards)

            '''5.3 Finding the ratio (pi_theta / pi_theta__old)'''
            ratios = torch.exp(log_probs - old_log_probs.mean(dim=1).detach())

            '''5.4 Finding Surrogate Loss'''
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            '''5.5 final loss of clipped objective PPO'''
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy

            '''5.6 take gradient step'''
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        '''6. Copy new weights into old policy'''
        self.policy_old.load_state_dict(self.policy.state_dict())

    def PPO_info(self):
        print('agent name：', self.env_msg['name'])
        print('state_dim:', self.env_msg['state_dim'])
        print('action_dim:', self.env_msg['action_dim'])
        print('action_range:', self.env_msg['action_range'])

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.env_msg['action_dim']):
            a = min(max(action[i], -1), 1)
            maxa = self.env_msg['action_range'][i][1]
            mina = self.env_msg['action_range'][i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return np.array(linear_action)
