import pandas as pd
from common.common_func import *
from common.common_cls import *
from environment.config.xml_write import xml_cfg
import cv2 as cv

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
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
        """
        @param env:
        @param gamma:
        @param noise_clip:
        @param noise_policy:
        @param policy_delay:
        @param critic1_soft_update:
        @param critic2_soft_update:
        @param actor_soft_update:
        @param memory_capacity:
        @param batch_size:
        @param actor:
        @param target_actor:
        @param critic1:
        @param target_critic1:
        @param critic2:
        @param target_critic2:
        @param path:
        """
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

        self.noise_OU = OUActionNoise(mu=np.zeros(self.env.action_dim))
        self.noise_gaussian = GaussianNoise(mu=np.zeros(self.env.action_dim))
        self.update_network_parameters()

        self.episode = 0
        self.reward = 0

        self.save_episode = []          # 保存的每一个回合的回合数
        self.save_reward = []           # 保存的每一个回合的奖励
        self.save_time = []
        self.save_average_reward = []   # 保存的每一个回合的平均时间的奖励
        self.save_successful_rate = []
        self.save_step = []             # 保存的每一步的步数
        self.save_stepreward = []       # 保存的每一步的奖励

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
            mu_prime = mu + torch.tensor(self.noise_gaussian(sigma=sigma), dtype=torch.float).to(self.actor.device)  # action with gaussian noise
            # mu_prime = mu + torch.tensor(self.noise_OU(), dtype=torch.float).to(self.actor.device)             # action with OU noise
        self.actor.train()  # 切换回训练模式
        mu_prime_np = mu_prime.cpu().detach().numpy()
        return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

    def evaluate(self, state):
        self.target_actor.eval()
        t_state = torch.tensor(state, dtype=torch.float).to(self.target_actor.device)  # get the tensor of the state
        act = self.target_actor(t_state).to(self.target_actor.device)  # choose action
        return act.cpu().detach().numpy()

    def agent_evaluate(self, test_num: int = 5, show_per: int = 10):
        for _ in range(test_num):
            self.env.reset_random()
            while not self.env.is_terminal:
                cv.waitKey(1)
                self.env.current_state = self.env.next_state.copy()

                t_state = torch.tensor(self.env.current_state, dtype=torch.float).to(self.target_actor.device)  # get the tensor of the state
                mu = self.target_actor(t_state).to(self.target_actor.device)  # choose action
                mu = mu.cpu().detach().numpy()
                action = self.action_linear_trans(mu)
                self.env.step_update(action)  # 环境更新的action需要是物理的action
                if self.episode % show_per == 0:
                    self.env.show_dynamic_image(isWait=False)
        cv.destroyAllWindows()

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

        self.target_actor.eval()            # PI'
        self.target_critic1.eval()          # Q1'
        self.critic1.eval()                 # Q1
        self.target_critic2.eval()          # Q2'
        self.critic2.eval()                 # Q2

        target_actions = self.target_actor.forward(new_state).to(self.critic1.device)                   # a' = PI'(s')
        '''动作正则化'''
        action_noise = torch.clip(torch.tensor(self.action_regularization(sigma=self.noise_policy)), -self.noise_clip, self.noise_clip).to(self.critic1.device)
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
        target = torch.tensor(reward + self.gamma * torch.minimum(critic_value1_.squeeze(), critic_value2_.squeeze()) * done).to(self.critic1.device)

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
                if random.randint(1, 2) == 1:
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
                target_param.data.copy_(target_param.data * (1.0 - self.critic1_tau) + param.data * self.critic1_tau)  # soft update
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.critic2_tau) + param.data * self.critic2_tau)  # soft update

        if self.policy_delay_iter % self.policy_delay == 0:
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update
            if is_target_critics_delay:
                for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.critic1_tau) + param.data * self.critic1_tau)  # soft update
                for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.critic2_tau) + param.data * self.critic2_tau)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def save_models_all(self):
        self.actor.save_all_net()
        self.critic1.save_all_net()
        self.critic2.save_all_net()
        self.target_actor.save_all_net()
        self.target_critic1.save_all_net()
        self.target_critic2.save_all_net()

    def load_models(self, path):
        """
        :brief:         only for test
        :param path:    file path
        :return:
        """
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(path + 'Actor_ddpg'))
        self.target_actor.load_state_dict(torch.load(path + 'TargetActor_ddpg'))
        self.critic1.load_state_dict(torch.load(path + 'Critic1_ddpg'))
        self.target_critic1.load_state_dict(torch.load(path + 'TargetCritic1_ddpg'))
        self.critic2.load_state_dict(torch.load(path + 'Critic2_ddpg'))
        self.target_critic2.load_state_dict(torch.load(path + 'TargetCritic2_ddpg'))

    def load_actor_optimal(self, path, file):
        print('...loading optimal...')
        self.actor.load_state_dict(torch.load(path + file))

    def load_target_actor_optimal(self, path, file):
        print('...loading optimal...')
        self.target_actor.load_state_dict(torch.load(path + file))

    def get_RLBase_from_XML(self, filename):
        rl_base, agentName = self.load_rl_basefromXML(filename=filename)
        state_dim_nn = int(rl_base['state_dim'])  # input dimension of NN
        action_dim_nn = int(rl_base['action_dim'])
        action_range = str2list(rl_base['action_range'])
        return agentName, state_dim_nn, action_dim_nn, action_range

    @staticmethod
    def load_rl_basefromXML(filename: str) -> (dict, str):
        """
        :brief:             从模型文件中加载数据到DQN中
        :param filename:    模型文件
        :return:            数据字典
        """
        root = xml_cfg().XML_Load(filename)
        return xml_cfg().XML_GetTagValue(node=xml_cfg().XML_FindNode(nodename='RL_Base', root=root)), root.attrib['name']

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

    def saveData_Step_Reward(self, step, reward, is2file=False, filename='StepReward.csv'):
        if is2file:
            data = pd.DataFrame({
                'step:': self.save_step,
                'stepreward': self.save_stepreward,
            })
            data.to_csv(self.path + filename, index=False, sep=',')
        else:
            self.save_step.append(step)
            self.save_stepreward.append(reward)

    def saveData_EpisodeReward(self, episode, time, reward, average_reward, successrate, is2file=False, filename='EpisodeReward.csv'):
        if is2file:
            data = pd.DataFrame({
                'episode': self.save_episode,
                'time': self.save_time,
                'reward': self.save_reward,
                'average_reward': self.save_average_reward,
                'success_rate': self.save_successful_rate
            })
            data.to_csv(self.path + filename, index=False, sep=',')
        else:
            self.save_episode.append(episode)
            self.save_time.append(time)
            self.save_reward.append(reward)
            self.save_average_reward.append(average_reward)
            self.save_successful_rate.append(successrate)
