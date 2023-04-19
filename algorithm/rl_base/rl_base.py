class rl_base:
    def __init__(self):
        self.state_dim = 0
        """
        The dimension of the state, which must be a finite number.
        For example, the dimension of a 2D UAV (mass point) is [px py vx vy ax ay],
        the 'state_dim' should be 6, but the number of each dimension can be finite (discrete) of infinite (continuous).
        Another example, the dimension of an inverted pendulum is [x, dx, ddx, theta, dtheta, ddtheta],
        the 'state_dim' should be 6.
        """

        self.state_num = []
        """
        The number of each state.
        It is a one dimensional list which includes the number of the state of each dimension.
        For example, for the inverted pendulum model we mentioned before:
        the 'state_num' should be [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        Another example, if we set the maximum of x as 10, minimum of x as -10, and the step of x as 2 and keep the other states unchanged,
        then we have: state_num = [11, math.inf, math.inf, math.inf, math.inf, math.inf].
        """

        self.state_step = []
        """
        It is a one dimensional list which includes the step of each state.
        Set the corresponding dimension as None if the state of which is a continuous one.
        If we set the maximum of x as 10, minimum of x as -10, and the step of x as 2 in the inverted pendulum example, 
        the state_step should be [2]; other wise, it should be [None]
        """

        self.state_space = []
        """
        It is a two dimensional list which includes all dimensions of states.
        Set the corresponding dimension as None if the state of which is a continuous one.
        If we set the maximum of x as 10, minimum of x as -10, and the step of x as 5 in the inverted pendulum example, 
        the state_space should be [[-10, -5, 0, 5, -10]]; other wise, it should be [[None]]
        """

        self.isStateContinuous = []
        """
        Correspondingly,
        the 'isStateContinuous' of inverted pendulum model is [True, True, True, True, True, True], or we can just set it to [True] 
        However, if we discrete the 'x' in the inverted pendulum model, then isStateContinuous = [False, True, True, True, True, True], 
        and we cannot simply set it to [True] or [False].
        """
        '''
        Generally speaking, the continuity of different states is identical.
        Because it is meaningless to make the problem more complicated deliberately without any benefit.
        '''

        self.action_dim = 0
        """
        The dimension of the action, which must be a finite number.
        For example, the dimension of a 2D UAV (mass point) is [px py vx vy ax ay],
        the 'action_dim' should be 2, which are the jerks of X and Y ([jx jy]).
        Another example, the dimension of an inverted pendulum is [x, dx, ddx, theta, dtheta, ddtheta],
        the 'action_dim' should be 1, which is the acceleration or jerk of the base.
        """

        self.action_num = []
        """
        The number of each action.
        It is a one dimensional list which includes the number of the action of each dimension.
        For example, for the inverted pendulum model we mentioned before:
        the 'action_num' should be [np.inf] if the acceleration is continuous.
        Another example, if we set the maximum of acceleration as 2, minimum of acceleration as -2, and the step as 0.5,
        then we have: action_num = [9], which are: 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0.
        """

        self.action_step = []
        """
        It is a one dimensional list which includes the step of each action.
        Set the corresponding dimension as None if the action of which is a continuous one.
        If we set the step of the acceleration of the base in the inverted pendulum as 1, 
        then the action_step should be [1]; other wise, it should be [None]
        """

        self.action_space = []
        """
        It is a two dimensional list which includes all dimensions of action.
        Set the corresponding dimension as None if the action of which is a continuous one.
        If we set the step of the acceleration of the base in the inverted pendulum as 1, minimum value as -2, and maximum value as 2
        the action_space should be [[-2, -1, 0, 1, 2]]; other wise, it should be [[None]]
        """

        self.isActionContinuous = []
        """
        If a DQN-related algorithm is implemented, then 'isActionContinuous' should be False due to the requirement of the RL algorithm.
        If a DDPG-related algorithm is implemented, then 'isActionContinuous' should be True due to the requirement of the RL algorithm.
        The 'isActionContinuous' is defined as a list because the dimension of the action may be larger than one,
        but the attributes of each dimension should be identical unless we insist on doing something 'strange'. For example:
        for a two-wheel ground vehicle, and we insist on only discretizing the torque of the left wheel,
        although unreasonably, the 'isActionContinuous' should be [False, True].
        """

        self.state_range = []
        """
        Formation: [[min1, max1], [min2, max2], ..., [minX, maxX]]
        It is a two dimensional list which includes the minimum and the maximum of each state.
        For example, if the state of a magnet levitation ball system is [x, v]. Then the state_range should be:
        [[-5, 5], [-15, 15]], which means the maximum of the position of the ball is 5, the minimum of the position of the ball is -5,
        the maximum of the velocity of the ball is 15, the minimum of the velocity of the ball is -15.
        But if we don't have a limitation of the velocity, the 'state_range' should be [[-5, 5], [-math.inf, math.inf]]
        """

        self.action_range = []
        """
        Formation: [[min1, max1], [min2, max2], ..., [minX, maxX]]
        It is a two dimensional list which includes the minimum and the maximum of each action.
        For example, if the action of a 3D UAV system is [ax, ay, az]. Then the action_range should be:
        [[-5, 5], [-5, 5], [-2, 2]], which means the maximum of the acceleration of the UAV is 5, 5, and -2 in the direction of X, Y, Z,
        the minimum of the acceleration of the UAV -5, -5, -2, respectively.
        Generally speaking, the range should not be infinite although it is mathematical-reasonable.
        """

        self.initial_state = []
        self.initial_action = []
        self.current_state = []
        self.next_state = []
        self.current_action = []
        self.reward = 0.0
        self.is_terminal = False

    def state_normalization(self, state: list, gain: float = 1.0, index0: int = -1, index1: int = -1):
        """
        :brief:             default for [-gain, gain]
        :param state:       state
        :param gain:        gain
        :param index1:
        :param index0:
        :return:            normalized state
        """
        length = len(state)
        # assert length == self.state_dim
        # assert length >= index1 >= index0
        start = 0 if index0 <= 0 else index0
        end = length - 1 if index1 > length - 1 else index1
        while start <= end:
            bound = self.state_range[start]
            k = 2 / (bound[1] - bound[0])
            b = 1 - bound[1] * k
            state[start] = (k * state[start] + b) * gain
            start += 1

    def step_update(self, action):
        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def get_reward(self, param):
        """
        :param param:       other parameters
        :return:            reward function
        """
        '''should be the function of current state, time, or next state. It needs to be re-written in a specific environment.'''
        pass

    def is_Terminal(self, param):
        return False

    def reset(self):
        # self.current_state = self.initial_state.copy()
        # self.next_state = []
        # self.reward = 0.0
        # self.is_terminal = False
        pass

    def reset_random(self):
        # self.current_state = []
        # for i in range(self.state_dim):
        #     if self.isStateContinuous[i]:
        #         if self.state_range[i][0] == -math.inf or self.state_range[i][1] == math.inf:
        #             self.current_state.append(0.0)
        #         else:
        #             self.current_state.append(random.uniform(self.state_range[i][0], self.state_range[i][1]))
        #     else:
        #         '''如果状态离散'''
        #         self.current_state.append(random.choice(self.state_space[i]))
        #
        # self.next_state = []
        # self.reward = 0.0
        # self.is_terminal = False
        pass
