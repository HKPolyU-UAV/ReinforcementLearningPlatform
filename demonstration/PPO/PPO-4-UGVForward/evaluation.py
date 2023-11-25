import datetime
import os
import sys
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from UGVForward import UGVForward as env
from utils.classes import *

optPath = './datasave/net/'
show_per = 1
timestep = 0
ENV = 'PPO-UGVForward'


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# setup_seed(2162)


class PPOActorCritic(nn.Module):
    def __init__(self, _state_dim, _action_dim, _action_range):
        super(PPOActorCritic, self).__init__()
        self.state_dim = _state_dim
        self.action_dim = _action_dim
        self.action_range = _action_range

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

    def forward(self):
        raise NotImplementedError

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            action_mean = self.actor(t_state)
        return action_mean.detach()

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.action_dim):
            a = min(max(action[i], -1), 1)
            maxa = self.action_range[i][1]
            mina = self.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return np.array(linear_action)


if __name__ == '__main__':
    log_dir = './datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulation_path = log_dir + datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
    os.mkdir(simulation_path)

    env = env()
    policy = PPOActorCritic(env.state_dim, env.action_dim, env.action_range)
    policy.load_state_dict(torch.load(optPath + 'actor-critic'))
    test_num = 3
    r = 0
    video = cv.VideoWriter('../PPO-4-' + env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 200,
                           (env.image_size[0] - env.board, env.image_size[1]))
    for i in range(test_num):
        r = 0
        env.reset(random=True)
        while not env.is_terminal:
            env.current_state = env.next_state.copy()
            _action_from_actor = policy.evaluate(env.current_state)
            _action = policy.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将actor输出动作转换到实际动作范围
            env.step_update(_action)  # 环境更新的动作必须是实际物理动作
            r += env.reward
            env.visualization()
            video.write(env.image[:, 0:env.image_size[0] - env.board])
        print(r)
    video.release()
