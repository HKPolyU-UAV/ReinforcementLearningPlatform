from common.common_cls import *
from algorithm.value_base.DQN import DQN
import torch.nn as nn
import torch

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
"""use CPU or GPU"""


class Dueling_DQN(DQN):
    def __init__(self,
                 env,
                 gamma:float,
                 epsilon: float,
                 learning_rate: float,
                 memory_capacity: int,
                 batch_size : int,
                 target_replace_iter: int,
                 eval_net: DuelingNeuralNetwork,
                 target_net: DuelingNeuralNetwork):
        super(Dueling_DQN, self).__init__(env, gamma, epsilon, learning_rate, memory_capacity, batch_size, target_replace_iter)

        '''Re-define NN'''
        self.device = device
        self.eval_net = eval_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        '''Re-define NN'''

    def DuelingDQN_info(self):
        print('This is Dueling DQN:')
        self.DQN_info()
