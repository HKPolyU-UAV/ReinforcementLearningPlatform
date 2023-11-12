import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from CartPole import CartPole
from algorithm.policy_base.Distributed_PPO import Distributed_PPO as DPPO
from utils.classes import *

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'DPPO-CartPole'


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(3407)


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

    env = CartPole(0, 0, False)

    agent = DPPO(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulationPath)
    agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1, 'GlobalPolicy',
                                         simulationPath)
    agent.load_models(optPath + 'actor-critic')
    # agent.global_policy.load_state_dict(torch.load(""))
    agent.eval_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, 0.1, 'EvalPolicy',
                                       simulationPath)
    agent.eval_policy.load_state_dict(agent.global_policy.state_dict())
    test_num = 5
    thetaError = []
    xError = []
    # cap = cv.VideoWriter('record.mp4', cv.VideoWriter_fourcc(*'mp4v'), 120,
    #                      (env.width, env.height))
    for _ in range(test_num):
        env.reset(random=True)
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            action_from_actor = agent.evaluate(env.current_state)
            action_from_actor = action_from_actor.numpy()
            action = agent.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
            env.step_update(action)  # 环境更新的action需要是物理的action
            env.visualization()  # 画图
            # cap.write(env.image[:, 0:env.width])
        thetaError.append(env.etheta)
        xError.append(env.ex)
    # cap.release()
    print(np.mean(thetaError), " ", np.mean(xError))
