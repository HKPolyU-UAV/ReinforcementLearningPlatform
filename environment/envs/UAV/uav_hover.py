import numpy as np

from environment.config.xml_write import *
from environment.envs.UAV.uav import UAV
from environment.envs.UAV.uav_visualization import UAV_Visualization
from algorithm.rl_base.rl_base import rl_base
from common.common_func import *
import pandas as pd


class UAV_Hover(rl_base, UAV):
    def __init__(self,
                 pos0=None,
                 vel0=None,
                 angle0=None,
                 omega0_inertial=None,
                 omega0_body=None,
                 target_pos=None,
                 save_cfg=True):
        UAV.__init__(self)

        self.init_pos = np.array([0, 0, 0]) if pos0 is None else np.array(pos0)
        self.init_vel = np.array([0, 0, 0]) if pos0 is None else np.array(vel0)
        self.init_angle = np.array([0, 0, 0]) if pos0 is None else np.array(angle0)
        self.init_omega0_inertial = np.array([0, 0, 0]) if pos0 is None else np.array(omega0_inertial)
        self.init_omega0_body = np.array([0, 0, 0]) if pos0 is None else np.array(omega0_body)
        self.target_pos = np.array(target_pos) if target_pos else self.init_pos.copy()

        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.angle = self.init_angle.copy()
        self.omega_inertial = self.init_omega0_inertial.copy()
        self.omega_body = self.init_omega0_body.copy()

        self.error_pos = self.target_pos - self.pos  # 位置跟踪误差
        self.staticGain = 2

        '''rl_base'''
        self.state_dim = 3 + 3 + 3 + 3 + 3  # ex ey ez x y z vx vy vz phi theta psi dphi dtheta dpsi
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 4
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = [[self.fmin, self.fmax] for _ in range(self.action_dim)]
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.Q_pos = np.diag([1, 1, 1])     # 用于位置的惩罚函数，若作为奖励，则需要 -e^T Q_pos e
        self.R = np.array([1, 1, 1, 1])     # 用于动作的惩罚函数，若作为奖励，则需要 -U(u)
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞障碍物
        '''rl_base'''

        '''visualization_opencv'''
        self.uav_vis = UAV_Visualization(xbound=np.array([self.pos_min[0], self.pos_max[0]]),
                                         ybound=np.array([self.pos_min[1], self.pos_max[1]]),
                                         zbound=np.array([self.pos_min[2], self.pos_max[2]]),
                                         origin=self.init_pos,
                                         target=self.target_pos)  # visualization
        '''visualization_opencv'''

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''
        if save_cfg:
            self.saveModel2XML()

    def state_norm(self) -> np.ndarray:
        """
        状态归一化
        """
        norm_error = self.error_pos / (self.pos_max - self.pos_min) * self.staticGain
        norm_pos = (2 * self.pos - self.pos_max - self.pos_min) / (self.pos_max - self.pos_min) * self.staticGain
        norm_vel = (2 * self.vel - self.vel_max - self.vel_min) / (self.vel_max - self.vel_min) * self.staticGain
        norm_angle = (2 * self.angle - self.angle_max - self.angle_min) / (self.angle_max - self.angle_min) * self.staticGain
        norm_dangle = (2 * self.omega_inertial - self.dangle_max - self.dangle_min) / (self.dangle_max - self.dangle_min) * self.staticGain
        # norm_f = (2 * self.force - self.fmax - self.fmin) / (self.fmax - self.fmin) * self.staticGain

        # norm_state = np.concatenate((norm_error, norm_pos, norm_vel, norm_angle, norm_dangle, norm_f)).tolist()
        norm_state = np.concatenate((norm_error, norm_pos, norm_vel, norm_angle, norm_dangle))

        return norm_state

    def inverse_state_norm(self, error_flag=False, pos_flag=False, vel_flag=False, angle_flag=False, dangle_flag=False):
        """
        状态反归一化
        """
        np_cur_state = np.array(self.current_state)
        _s = np.array([])
        if error_flag:
            _s = np.concatenate((_s, np_cur_state[0:3] / self.staticGain * (self.pos_max - self.pos_min)))
        if pos_flag:
            _s = np.concatenate((_s, (np_cur_state[3:6] / self.staticGain * (self.pos_max - self.pos_min) + self.pos_max + self.pos_min) / 2))
        if vel_flag:
            _s = np.concatenate((_s, (np_cur_state[6:9] / self.staticGain * (self.vel_max - self.vel_min) + self.vel_max + self.vel_min) / 2))
        if angle_flag:
            _s = np.concatenate((_s, (np_cur_state[9:12] / self.staticGain * (self.angle_max - self.angle_min) + self.angle_max + self.angle_min) / 2))
        if dangle_flag:
            _s = np.concatenate((_s, (np_cur_state[12:15] / self.staticGain * (self.dangle_max - self.dangle_min) + self.dangle_max + self.dangle_min) / 2))
        # if f_flag:
        #     _s = np.concatenate((_s, (np_cur_state[15:19] / self.staticGain * (self.fmax - self.fmin) + self.fmax + self.fmin) / 2))

        return _s

    def saveModel2XML(self, filename='UAV_Hover.xml', filepath='../../environment/config/'):
        rootMsg = {
            'name': 'uav_hover',
            'author': 'Yefeng YANG',
            'date': '2023.02.27',
            'E-mail': 'yefeng.yang@connect.polyu.hk; 18B904013@stu.hit.edu.cn'
        }
        rl_baseMsg = {
            'state_dim': self.state_dim,
            'state_num': self.state_num,
            'state_step': self.state_step,
            'state_space': self.state_space,
            'state_range': self.state_range,
            'isStateContinuous': self.isStateContinuous,
            'initial_state': self.initial_state,
            'current_state': self.current_state,
            'next_state': self.next_state,
            'action_dim': self.action_dim,
            'action_step': self.action_step,
            'action_range': self.action_range,
            'action_num': self.action_num,
            'action_space': self.action_space,
            'isActionContinuous': self.isActionContinuous,
            'initial_action': self.initial_action,
            'current_action': self.current_action,
            'is_terminal': self.is_terminal,
            'staticGain': self.staticGain
        }
        physicalMsg = {
            'm': self.m,
            'g': self.g,
            'Jxx': self.Jxx,
            'Jyy': self.Jyy,
            'Jzz': self.Jzz,
            'd': self.d,
            'CT': self.CT,
            'CM': self.CM,
            'J0': self.J0,
            'pos': self.pos,
            'vel': self.vel,
            'angle': self.angle,
            'omega_iner': self.omega_inertial,
            'omega_body': self.omega_body,
            'pos_max': self.pos_max,
            'pos_min': self.pos_min,
            'vel_max': self.vel_max,
            'vel_min': self.vel_min,
            'angle_max': self.angle_max,
            'angle_min': self.angle_min,
            'dangle_max': self.dangle_max,
            'dangle_min': self.dangle_min,
            'force': self.force,
            'fmax': self.fmax,
            'fmin': self.fmin,
            'w_rotor': self.w_rotor,
            'dt': self.dt,
            'time': self.time,
            'tmax': self.tmax,
            'targetpos': self.target_pos
        }
        imageMsg = {
            'figsize': [9, 9],
            'xbound': self.uav_vis.xbound,
            'ybound': self.uav_vis.ybound,
            'zbound': self.uav_vis.zbound,
            'length_per_n': self.uav_vis.length_per_n
        }
        xml_cfg().XML_Create(filename=filepath + filename,
                             rootname='Plant',
                             rootmsg=rootMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='RL_Base',
                                 nodemsg=rl_baseMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Physical',
                                 nodemsg=physicalMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Image',
                                 nodemsg=imageMsg)
        xml_cfg().XML_Pretty_All(filepath + filename)

    def saveData(self, is2file=False, filename='uav.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.save_pos[:, 0],
                'y': self.save_pos[:, 1],
                'z': self.save_pos[:, 2],
                'vx': self.save_vel[:, 0],
                'vy': self.save_vel[:, 1],
                'xz': self.save_vel[:, 2],
                'phi': self.save_angle[:, 0],
                'theta': self.save_angle[:, 1],
                'psi': self.save_angle[:, 2],
                'dphi': self.save_omega_inertial[:, 0],
                'dtheta': self.save_omega_inertial[:, 1],
                'dpsi': self.save_omega_inertial[:, 2],
                'p': self.save_omega_body[:, 0],
                'q': self.save_omega_body[:, 1],
                'r': self.save_omega_body[:, 2],
                'f1': self.save_f[:, 0],
                'f2': self.save_f[:, 1],
                'f3': self.save_f[:, 2],
                'f4': self.save_f[:, 3],
                'time': self.save_t
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_pos = np.vstack((self.save_pos, self.pos))
            self.save_vel = np.vstack((self.save_vel, self.vel))
            self.save_angle = np.vstack((self.save_angle, self.angle))
            self.save_omega_inertial = np.vstack((self.save_omega_inertial, self.omega_inertial))
            self.save_omega_body = np.vstack((self.save_omega_body, self.omega_body))
            self.save_f = np.vstack((self.save_f, self.force))
            self.save_t = np.vstack((self.save_t, self.time))

    def get_reward(self, param=None):
        """
        @note:              reward function
        @param param:       extra parameters
        @return:            None, but update the reward
        """
        ss = self.inverse_state_norm(error_flag=True, pos_flag=False, vel_flag=False, angle_flag=False, dangle_flag=False)
        '''
        这个函数是把归一化的状态还原，其实这个操作是不必要的。因为所有状态本身是已知的，确实多此一举，如此操作只是为了形式上的统一
        目前奖励函数只使用了位置误差 和 输出力，所以error_flag 和 f_flag 是True，其余都是False
        所以，ss中一共有7个数，前三个是位置误差，后四个是力
        '''
        _error = ss[0:3]
        _force = self.force
        # self.force = _force     # 四个螺旋桨的力
        # self.f2omega()          # 得到四个电机的转速
        # virtual_input = np.dot(self.power_allocation_mat, self.w_rotor ** 2)    # 得到实际的虚拟控制，油门 三个方向转矩

        '''位置误差奖励'''
        r_reward = -np.dot(np.dot(_error, self.Q_pos), _error)
        '''位置误差奖励'''

        '''控制奖励'''
        # 这里的奖励实际上是以惩罚的形式给出的，所以需要负号
        _lambda = self.fmax
        if np.min(np.fabs(_lambda - _force)) < 1e-2:      # 说明至少一个力饱和输出
            u_reward = -np.dot((_lambda + _force) * np.log(_lambda + _force)
                               - 2 * _lambda * np.log(_lambda), self.R)
        else:
            u_reward = -np.dot((_lambda + _force) * np.log(_lambda + _force)
                               + (_lambda - _force) * np.log(_lambda - _force)
                               - 2 * _lambda * np.log(_lambda), self.R)

        '''控制奖励'''

        self.reward = r_reward + u_reward

    def is_success(self):
        """
        @note:      to judge if a uav has arrived to the target successfully
        @return:    yes or no
        """
        if np.linalg.norm(self.error_pos) < 20e-2:   # 位置到终点
            if np.linalg.norm(self.vel) < 3e-2:     # 并且不能有速度
                return True
        return False

    def is_Terminal(self, param=None):
        if self.is_success():
            print('Very good!!!')
            self.terminal_flag = 3
            self.is_terminal = True
            return True
        return self.is_episode_Terminal()

    def step_update(self, action: list):
        """
        @note:          state update, both state in RL and model
        @param action:  action (control input, command)
        @return:        current_state, action, reward, next_state, is_terminal (all in RL)
        """
        self.current_action = action.copy()
        self.current_state = self.state_norm()      # 当前状态

        '''rk44'''
        self.rk44(action=np.array(action))
        self.error_pos = self.target_pos - self.pos  # 非常关键, sun of a bitch, 我给忘了
        '''rk44'''

        self.is_terminal = self.is_Terminal()
        self.next_state = self.state_norm()
        self.get_reward()

        '''出界处理'''
        # 暂无
        '''出界处理'''

        self.saveData()

    def set_random_pos(self):
        """
        @note:      set the initial position randomly
        @return:    initial position
        """
        # 必须保证初始化的位置与边界至少保证一个无人机的空隙
        if np.min(self.pos_max - self.pos_min) < 6 * self.d:
            # 如果课活动空间本身太小，那么直接返回原来的位置
            return self.init_pos
        pr = self.pos_max - 3 * self.d
        pl = self.pos_min + 3 * self.d
        return np.array([random.uniform(pl[0], pr[0]),
                         random.uniform(pl[1], pr[1]),
                         random.uniform(pl[2], pr[2])])

    def set_random_target_pos(self):
        """
        @note:      set target randomly
        @return:    target position
        """
        # 必须保证初始化的位置与边界至少保证一个无人机的空隙
        if np.min(self.pos_max - self.pos_min) < 6 * self.d:
            # 如果课活动空间本身太小，那么直接返回原来的位置
            return self.init_pos
        pr = self.pos_max - 3 * self.d
        pl = self.pos_min + 3 * self.d
        return np.array([random.uniform(pl[0], pr[0]),
                         random.uniform(pl[1], pr[1]),
                         random.uniform(pl[2], pr[2])])

    def show_dynamic_image(self, per_show=10):
        """
        @note:              show image
        @param per_show:    draw the graph every 'per_show' time steps
        @return:            None
        """
        self.uav_vis.render(p=self.pos, ref_p=self.target_pos, v=self.vel, a=self.angle, ra=self.omega_inertial, d=self.d, f=self.force, win=per_show)

    def reset(self):
        """
        @note:      reset the environment
        @return:    None
        """
        '''physical parameters'''
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.angle = self.init_angle.copy()
        self.omega_inertial = self.init_omega0_inertial.copy()
        self.omega_body = self.init_omega0_body.copy()
        self.error_pos = self.target_pos - self.pos  # 位置跟踪误差
        self.time = 0
        self.control_state = np.concatenate((self.pos, self.vel, self.angle, self.omega_inertial))  # 控制系统的状态，不是强化学习的状态
        self.force = np.array([0, 0, 0, 0])
        self.w_rotor = np.sqrt(self.force / self.CT)
        self.uav_vis.reset(self.init_pos, self.target_pos)
        '''physical parameters'''

        '''rl_base'''
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时
        '''rl_base'''

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''

    def reset_random(self):
        """
        @note:      reset the environment randomly (just the target and initial position)
        @return:    None
        """
        # 这里仅仅是位置random，初始的速度都是零
        '''physical parameters'''
        self.pos = self.set_random_pos()
        self.target_pos = self.set_random_target_pos()
        self.init_pos = self.pos.copy()
        self.vel = self.init_vel.copy()
        self.angle = self.init_angle.copy()
        self.omega_inertial = self.init_omega0_inertial.copy()
        self.omega_body = self.init_omega0_body.copy()
        self.error_pos = self.target_pos - self.pos  # 位置跟踪误差
        self.time = 0
        self.control_state = np.concatenate((self.pos, self.vel, self.angle, self.omega_inertial))  # 控制系统的状态，不是强化学习的状态
        self.force = np.array([0, 0, 0, 0])
        self.w_rotor = np.sqrt(self.force / self.CT)
        self.uav_vis.reset(self.init_pos, self.target_pos)
        '''physical parameters'''

        '''rl_base'''
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时
        '''rl_base'''

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''

    def reset_target_random(self):
        """
        @note:      reset the environment randomly (just the target)
        @return:    None
        """
        # 这里仅仅是位置random，初始的速度都是零
        '''physical parameters'''
        # self.pos = self.set_random_pos()
        self.target_pos = self.set_random_target_pos()
        self.pos = self.init_pos.copy()
        self.vel = self.init_vel.copy()
        self.angle = self.init_angle.copy()
        self.omega_inertial = self.init_omega0_inertial.copy()
        self.omega_body = self.init_omega0_body.copy()
        self.error_pos = self.target_pos - self.pos  # 位置跟踪误差
        self.time = 0
        self.control_state = np.concatenate((self.pos, self.vel, self.angle, self.omega_inertial))  # 控制系统的状态，不是强化学习的状态
        self.force = np.array([0, 0, 0, 0])
        self.w_rotor = np.sqrt(self.force / self.CT)
        self.uav_vis.reset(self.init_pos, self.target_pos)
        '''physical parameters'''

        '''rl_base'''
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时
        '''rl_base'''

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''