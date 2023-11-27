import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from utils.classes import PPOActor_Gaussian, PPOCritic, SharedAdam, RolloutBuffer, Normalization


class Worker(mp.Process):
    def __init__(self,
                 g_actor: PPOActor_Gaussian,
                 l_actor: PPOActor_Gaussian,
                 g_critic: PPOCritic,
                 l_critic: PPOCritic,
                 g_opt_critic: SharedAdam,
                 g_opt_actor: SharedAdam,
                 g_train_n: mp.Value,
                 _index: int,
                 _name: str,
                 _env,
                 _queue: mp.Queue,
                 _lock: mp.Lock,
                 _ppo_msg: dict):
        super(Worker, self).__init__()
        self.g_actor = g_actor
        self.l_actor = l_actor
        self.g_critic = g_critic
        self.l_critic = l_critic
        self.g_opt_critic = g_opt_critic
        self.g_opt_actor = g_opt_actor
        self.global_training_num = g_train_n
        self.index = _index
        self.name = _name
        self.env = _env
        self.queue = _queue
        self.lock = _lock
        self.buffer = RolloutBuffer(int(self.env.time_max / self.env.dt * 4), self.env.state_dim, self.env.action_dim)
        self.gamma = _ppo_msg['gamma']
        self.k_epo = _ppo_msg['k_epo']
        self.device = _ppo_msg['device']
        self.lmd = _ppo_msg['lmd']
        self.use_adv_norm = _ppo_msg['use_adv_norm']
        self.eps_clip = _ppo_msg['eps_clip']
        self.entropy_coef = _ppo_msg['entropy_coef']
        self.use_grad_clip = _ppo_msg['use_grad_clip']
        self.use_lr_decay = _ppo_msg['use_lr_decay']
        self.ppo_msg = _ppo_msg
        self.episode = 0

    def learn(self):
        """
		@note: 	 network update
		@return: None
		"""
        '''前期数据处理'''
        s, a, a_lp, r, s_, done, success = self.buffer.to_tensor()

        adv = []
        gae = 0.
        with torch.no_grad():
            vs = self.l_critic.net(s)  # TODO YYF 也改了
            vs_ = self.l_critic.net(s_)  # TODO YYF 也改了
            deltas = r + self.gamma * (1.0 - success) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lmd * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.k_epo):  # 每次的轨迹数据学习 K_epochs 次
            dist_now = self.l_actor.get_dist(s)
            dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
            a_logprob_now = dist_now.log_prob(a)

            # Update actor
            ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_lp.sum(1, keepdim=True))
            surr1 = ratios * adv  # Only calculate the gradient of 'a_logprob_now' in ratios
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
            self.g_opt_actor.zero_grad()
            actor_loss.mean().backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.l_actor.parameters(), 0.5)  # TODO YYF 也改了
            for l_a, g_a in zip(self.l_actor.parameters(), self.g_actor.parameters()):
                g_a._grad = l_a.grad
            self.g_opt_actor.step()

            # Update critic
            v_s = self.l_critic.net(s)
            critic_loss = F.mse_loss(v_target, v_s)
            self.g_opt_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.l_critic.parameters(), 0.5)  # TODO YYF 也改了
            for l_c, g_c in zip(self.l_critic.parameters(), self.g_critic.parameters()):
                g_c._grad = l_c.grad
            self.g_opt_critic.step()

    def choose_action(self, state: np.ndarray):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
            dist = self.l_actor.get_dist(t_state)
            a = dist.sample()
            a = torch.maximum(torch.minimum(a, self.l_actor.a_max), self.l_actor.a_min)  # TODO a_max a_min 正确
            a_logprob = dist.log_prob(a)
        return a.detach().cpu().numpy().flatten(), a_logprob.detach().cpu().numpy().flatten()

    def run(self):
        buffer_index = 0
        sumr = 0.
        t_epoch = 0
        timestep = 0
        reward_norm = Normalization(shape=1)
        while True:
            self.l_actor.load_state_dict(self.g_actor.state_dict())
            self.l_critic.load_state_dict(self.g_critic.state_dict())
            '''1. 收集数据'''
            while buffer_index < self.buffer.batch_size:
                if self.env.is_terminal:  # 如果某一个回合结束
                    # print('Sumr:  ', sumr)
                    sumr = 0.
                    self.env.reset(True)
                else:
                    self.env.current_state = self.env.next_state.copy()
                    a, a_log_prob = self.choose_action(self.env.current_state)
                    self.env.step_update(a)
                    # env.visualization()
                    sumr += self.env.reward
                    if self.env.is_terminal:
                        if self.env.terminal_flag == 2:
                            success = 0
                        else:  # 只有回合结束，并且过早结束的时候，才是 1
                            success = 1
                    else:
                        success = 0
                    # success = 1.0
                    self.buffer.append(s=self.env.current_state,
                                       a=a,
                                       log_prob=a_log_prob,
                                       r=reward_norm(self.env.reward),
                                       # r=self.env.reward,
                                       s_=self.env.next_state,
                                       done=1.0 if self.env.is_terminal else 0.0,
                                       success=success,
                                       index=buffer_index)
                    buffer_index += 1
            '''1. 收集数据'''

            '''2. 学习'''
            # print('~~~~~~~~~~ Training Start ~~~~~~~~~~')
            # print('Train Epoch: {}'.format(t_epoch))
            timestep += self.ppo_msg['buffer_size']
            self.learn()
            buffer_index = 0
            '''2. 学习'''

            '''4. 每学习 1000 次，减小一次探索概率'''
            if t_epoch % 1000 == 0 and t_epoch > 0:
                _ratio = max(1 - t_epoch / 1000 * 0.05, 0.2)
                self.l_actor.std *= _ratio
                print("setting actor output action_std to : ", self.l_actor.std)
            '''4. 每学习 1000 次，减小一次探索概率'''

            t_epoch += 1
            with self.lock:
                self.global_training_num.value += 1
            self.queue.put(round(sumr, 2))
        # print('~~~~~~~~~~  Training End ~~~~~~~~~~')
    # self.queue.put(None)  # 这个进程结束了，就把None放进去，用于global判断


class Distributed_PPO2:
    def __init__(self, env, actor_lr: float = 3e-4, critic_lr: float = 1e-3, num_of_pro: int = 5, path: str = ''):
        """
		@param env:			RL environment
		@param actor_lr:	actor learning rate
		@param critic_lr:	critic learning rate
		@param num_of_pro:	number of training process
		"""
        '''RL env'''
        # Remind: 这里的 env 还有 env 的成员函数是找不到索引的，需要确保写的时候没有问题才行，为了方便设计，不得已把 env 集成到 DPPO 里面
        # 如果形成一种习惯，也未尝不可，嗨嗨嗨
        self.env = env
        self.state_dim_nn = env.state_dim
        self.action_dim_nn = env.action_dim
        self.action_range = env.action_range
        '''RL env'''

        '''DPPO'''
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_of_pro = num_of_pro  # 进程数量
        self.queue = mp.Queue()
        self.global_training_num = mp.Value('i', 0)
        self.lock = mp.Lock()
        self.path = path
        '''DPPO'''

        '''global variable'''
        self.global_critic = PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True)
        self.global_actor = PPOActor_Gaussian(state_dim=env.state_dim,
                                              action_dim=env.action_dim,
                                              a_min=np.array(env.action_range)[:, 0],
                                              a_max=np.array(env.action_range)[:, 1],
                                              init_std=0.8,
                                              use_orthogonal_init=True)
        self.eval_actor = PPOActor_Gaussian(state_dim=env.state_dim,
                                            action_dim=env.action_dim,
                                            a_min=np.array(env.action_range)[:, 0],
                                            a_max=np.array(env.action_range)[:, 1],
                                            init_std=0.8,
                                            use_orthogonal_init=True)

        self.actor_optimizer = SharedAdam([{'params': self.global_actor.parameters(), 'lr': self.actor_lr}])
        self.critic_optimizer = SharedAdam([{'params': self.global_critic.parameters(), 'lr': self.critic_lr}])

        self.global_actor.share_memory()
        self.global_critic.share_memory()

        '''global variable'''
        self.device = 'cpu'
        '''multi process'''
        self.processes = [mp.Process(target=self.global_evaluate, args=())]  # evaluation process
        '''multi process'''

        self.evaluate_record = []
        self.training_record = []

    def save_ac(self, msg, path):
        torch.save(self.global_actor.state_dict(), path + 'actor' + msg)
        torch.save(self.global_critic.state_dict(), path + 'critic' + msg)

    def global_evaluate(self):
        while True:
            training_r = self.queue.get()
            if training_r is None:
                break
            if self.global_training_num.value % 50 == 0:
                print('Training count: ', self.global_training_num.value)

            if self.global_training_num.value % 500 == 0:
                self.eval_actor.load_state_dict(self.global_actor.state_dict())  # 复制 global policy

                training_num_temp = self.global_training_num.value  # 记录一下当前的数字，因为测试和学习同时进行的，号码容易窜
                print('...saving check point... ', int(training_num_temp))
                temp = self.path + 'trainNum_{}/'.format(training_num_temp)
                os.mkdir(temp)
                time.sleep(0.01)
                self.save_ac(msg='', path=temp)

                eval_num = 5
                for i in range(eval_num):
                    print('测试: ', i)
                    self.env.reset(True)
                    r = 0
                    while not self.env.is_terminal:
                        self.env.current_state = self.env.next_state.copy()
                        _a = self.evaluate(self.env.current_state)
                        # print(_a)
                        self.env.step_update(_a)
                        r += self.env.reward
                        self.env.visualization()
                    print('Test: %.2f' % (r))
                # cv.destroyAllWindows()
        print('...training end...')

    def add_worker(self, worker: Worker):
        self.processes.append(worker)

    def start_multi_process(self):
        for p in self.processes:
            p.start()
            p.join(0.5)
        # print('Finish loading all workers')

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            action_mean = self.eval_actor.evaluate(t_state)
        return action_mean

    def DPPO2_info(self):
        print('number of process:', self.num_of_pro)
        print('agent name：', self.env.name)
        print('state_dim:', self.state_dim_nn)
        print('action_dim:', self.action_dim_nn)
        print('action_range:', self.action_range)
