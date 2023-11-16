from collector import data_collector
from FNTSMC import fntsmc_att, fntsmc_param
from uav import UAV, uav_param
from ref_cmd import *


class uav_att_ctrl(UAV):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param):
        super(uav_att_ctrl, self).__init__(UAV_param)
        self.att_ctrl = fntsmc_att(att_ctrl_param)
        self.collector = data_collector(round(self.time_max / self.dt))
        self.ref = np.zeros(3)
        self.dot_ref = np.zeros(3)

    def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray):
        """
        @param ref:         参考信号
        @param dot_ref:     参考信号一阶导数
        @param dot2_ref:    参考信号二阶导数 (仅在姿态控制模式有效)
        @return:            Tx Ty Tz
        """
        self.ref = ref
        self.dot_ref = dot_ref
        e = self.rho1() - self.ref
        de = self.dot_rho1() - self.dot_ref
        sec_order_att_dy = self.second_order_att_dynamics()
        ctrl_mat = self.att_control_matrix()
        self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, dot2_ref)
        return self.att_ctrl.control

    def update(self, action: np.ndarray):
        """
        @param action:  三个力矩
        @return:
        """
        action_4_uav = np.insert(action, 0, 0)
        data_block = {'time': self.time,                    # simulation time
                      'control': action_4_uav,              # actual control command
                      'ref_angle': self.ref,                # reference angle
                      'ref_pos': np.zeros(3),               # set to zero for attitude control
                      'ref_vel': np.zeros(3),               # set to zero for attitude control
                      'd_out': np.zeros(3),                 # set to zero for attitude control
                      'd_out_obs': np.zeros(3),             # set to zero for attitude control
                      'state': self.uav_state_call_back()}  # quadrotor state
        self.collector.record(data_block)
        self.rk44(action=action_4_uav, dis=np.zeros(3), n=1, att_only=True)
