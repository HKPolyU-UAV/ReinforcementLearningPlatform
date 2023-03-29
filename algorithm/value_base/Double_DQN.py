from algorithm.value_base.DQN import DQN
import torch

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
"""use CPU or GPU"""


class Double_DQN(DQN):
    def __init__(self,
                 env,
                 gamma,
                 epsilon,
                 learning_rate,
                 memory_capacity,
                 batch_size,
                 target_replace_iter,
                 eval_net,
                 target_net):
        super(Double_DQN, self).__init__(env, gamma, epsilon, learning_rate, memory_capacity, batch_size, target_replace_iter, eval_net, target_net)

    def nn_training(self, saveNNPath=None, is_reward_ascent=False):
        """
        @param saveNNPath:
        @param is_reward_ascent:
        @return:
        """
        self.target_replace_count += 1
        if self.target_replace_count % self.target_replace_iter == 0:  # 满足这个条件，网络参数就更新一次
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.target_net, saveNNPath + '/' + 'dqn.pkl')
            torch.save(self.target_net.state_dict(), saveNNPath + '/' + 'dqn_parameters.pkl')
            torch.save(self.eval_net, saveNNPath + '/' + 'eval_dqn.pkl')
            torch.save(self.eval_net.state_dict(), saveNNPath + '/' + 'eval_dqn_parameters.pkl')
            print('...network update...', int(self.target_replace_count / self.target_replace_iter))

        state, action, reward, new_state, done = self.memory.sample_buffer()
        t_s = torch.tensor(state, dtype=torch.float).to(device)
        t_a_pos = self.torch_action2num(action).to(device)  # t_a是具体的物理动作，需要转换成动作编号作为索引值，是个tensor
        t_r = torch.tensor(reward, dtype=torch.float).to(device)
        t_s_ = torch.tensor(new_state, dtype=torch.float).to(device)
        t_bool = torch.tensor(done, dtype=torch.float).to(device)
        q_next = torch.squeeze(self.target_net(t_s_).detach().float()).to(device)

        '''Double DQN'''
        ddqn_action_value2 = self.eval_net(t_s_).detach()
        t_ddqn_num2 = torch.argmax(ddqn_action_value2, dim=1, keepdim=True)
        q_target = t_r + self.gamma * (torch.gather(q_next, 1, t_ddqn_num2).mul(t_bool))
        # print(t_ddqn_num2.size())
        '''Double DQN'''

        for _ in range(1):
            q_eval = self.eval_net(t_s).gather(1, t_a_pos)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saveData_StepTDErrorNNLose(self.target_replace_count,
                                            (q_target - q_eval).sum().detach().cpu().numpy(),
                                            loss.detach().cpu().numpy())

    def DoubleDQN_info(self):
        print('Double DQN agent name:', self.env.name)
        print('Double DQN input dimension:', self.env.state_dim)
        print('Double DQN output dimension:', self.env.action_num)
        print('Agent physical action dimension:', self.env.action_dim)
        print('Agent action space:', self.env.action_space)
        print('Replay memory capitaty:', self.memory_capacity)
        print('Batch size:', self.batch_size)
