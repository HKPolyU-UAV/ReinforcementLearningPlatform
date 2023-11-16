import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class data_collector:
    def __init__(self, N: int):
        self.N = N
        self.t = np.zeros((self.N, 1)).astype(float)
        self.control = np.zeros((self.N, 4)).astype(float)
        self.ref_angle = np.zeros((self.N, 3)).astype(float)
        self.ref_pos = np.zeros((self.N, 3)).astype(float)
        self.ref_vel = np.zeros((self.N, 3)).astype(float)
        self.d_out = np.zeros((self.N, 3)).astype(float)
        self.d_out_obs = np.zeros((self.N, 3)).astype(float)
        self.state = np.zeros((self.N, 12)).astype(float)
        self.index = 0
        self.name = ['uav_state.csv', 'ref_cmd.csv', 'control.csv', 'observe.csv']

    def record(self, data: dict):
        if self.index < self.N:
            self.t[self.index][0] = data['time']
            self.control[self.index] = data['control']
            self.ref_angle[self.index] = data['ref_angle']
            self.ref_pos[self.index] = data['ref_pos']
            self.ref_vel[self.index] = data['ref_vel']
            self.d_out[self.index] = data['d_out']
            self.d_out_obs[self.index] = data['d_out_obs']
            self.state[self.index] = data['state']
            self.index += 1

    def reset(self, N: int):
        self.N = N
        self.t = np.zeros((self.N, 1)).astype(float)
        self.control = np.zeros((self.N, 4)).astype(float)
        self.ref_angle = np.zeros((self.N, 3)).astype(float)
        self.ref_pos = np.zeros((self.N, 3)).astype(float)
        self.ref_vel = np.zeros((self.N, 3)).astype(float)
        self.d_out = np.zeros((self.N, 3)).astype(float)
        self.d_out_obs = np.zeros((self.N, 3)).astype(float)
        self.state = np.zeros((self.N, 12)).astype(float)
        self.index = 0

    def package2file(self, path: str):
        pd.DataFrame(np.hstack((self.t, self.state)),
                     columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']). \
            to_csv(path + self.name[0], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.ref_pos, self.ref_vel, self.ref_angle)),
                     columns=['time', 'ref_x', 'ref_y', 'ref_z', 'ref_vx', 'ref_vy', 'ref_vz', 'ref_phi', 'ref_theta', 'ref_psi']). \
            to_csv(path + self.name[1], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.control)),
                     columns=['time', 'throttle', 'torque_x', 'torque_y', 'torque_z']). \
            to_csv(path + self.name[2], sep=',', index=False)

        pd.DataFrame(np.hstack((self.t, self.d_out, self.d_out_obs)),
                     columns=['time', 'out1', 'out2', 'out3', 'out1_obs', 'out2_obs', 'out3_obs']). \
            to_csv(path + self.name[3], sep=',', index=False)

    def load_file(self, path: str):
        controlData = pd.read_csv(path + 'control.csv', header=0).to_numpy()
        observeData = pd.read_csv(path + 'observe.csv', header=0).to_numpy()
        ref_cmdData = pd.read_csv(path + 'ref_cmd.csv', header=0).to_numpy()
        uav_stateData = pd.read_csv(path + 'uav_state.csv', header=0).to_numpy()

        self.t = controlData[:, 0]
        self.control = controlData[:, 1: 5]
        self.ref_pos = ref_cmdData[:, 1: 4]
        self.ref_vel = ref_cmdData[:, 4: 7]
        self.ref_angle = ref_cmdData[:, 7: 10]
        self.d_out, self.d_out_obs = observeData[:, 1:4], observeData[:, 4:7]
        self.state = uav_stateData[:, 1: 13]

    def plot_pos(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_pos[:, 0], 'red')
        plt.plot(self.t, self.state[:, 0], 'blue')
        plt.grid(True)
        plt.ylim((-5, 5))
        plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('X')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_pos[:, 1], 'red')
        plt.plot(self.t, self.state[:, 1], 'blue')
        plt.grid(True)
        plt.ylim((-5, 5))
        plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('Y')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_pos[:, 2], 'red')
        plt.plot(self.t, self.state[:, 2], 'blue')
        plt.grid(True)
        plt.ylim((-5, 5))
        plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('Z')

    def plot_vel(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_vel[:, 0], 'red')
        plt.plot(self.t, self.state[:, 3], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_vel[:, 1], 'red')
        plt.plot(self.t, self.state[:, 4], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vy')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_vel[:, 2], 'red')
        plt.plot(self.t, self.state[:, 5], 'blue')
        plt.grid(True)
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 5, 1))
        plt.xlabel('time(s)')
        plt.title('vz')

    def plot_att(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.ref_angle[:, 0] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 6] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-90, 90))
        plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('roll-phi')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.ref_angle[:, 1] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 7] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-90, 90))
        plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('pitch-theta')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.ref_angle[:, 2] * 180 / np.pi, 'red')
        plt.plot(self.t, self.state[:, 8] * 180 / np.pi, 'blue')
        plt.grid(True)
        plt.ylim((-100, 100))
        plt.yticks(np.arange(-100, 100, 10))
        plt.xlabel('time(s)')
        plt.title('yaw-psi')

    def plot_pqr(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.state[:, 9] * 180 / np.pi, 'blue')
        plt.grid(True)
        # plt.ylim((-90, 90))
        # plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('p')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.state[:, 10] * 180 / np.pi, 'blue')
        plt.grid(True)
        # plt.ylim((-90, 90))
        # plt.yticks(np.arange(-90, 90, 10))
        plt.xlabel('time(s)')
        plt.title('q')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.state[:, 11] * 180 / np.pi, 'blue')
        plt.grid(True)
        # plt.ylim((-100, 100))
        # plt.yticks(np.arange(-100, 100, 10))
        plt.xlabel('time(s)')
        plt.title('r')

    def plot_throttle(self):
        plt.figure()
        plt.plot(self.t, self.control[:, 0], 'red')  # 油门
        plt.grid(True)
        plt.title('throttle')

    def plot_torque(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.control[:, 1], 'red')  # Tx
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        # plt.xlabel('time(s)')
        plt.title('Tx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.control[:, 2], 'red')  # Ty
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        plt.xlabel('time(s)')
        plt.title('Ty')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.control[:, 3], 'red')  # Tz
        plt.grid(True)
        # plt.ylim((-0.3, 0.3))
        # plt.yticks(np.arange(-0.3, 0.3, 0.1))
        plt.xlabel('time(s)')
        plt.title('Tz')

    def plot_outer_obs(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(self.t, self.d_out[:, 0], 'red')
        plt.plot(self.t, self.d_out[:, 0], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.ylim((-4, 4))
        plt.title('observe dx')

        plt.subplot(1, 3, 2)
        plt.plot(self.t, self.d_out[:, 1], 'red')
        plt.plot(self.t, self.d_out[:, 1], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.ylim((-4, 4))
        plt.title('observe dy')

        plt.subplot(1, 3, 3)
        plt.plot(self.t, self.d_out[:, 2], 'red')
        plt.plot(self.t, self.d_out[:, 2], 'blue')
        plt.grid(True)
        plt.xlabel('time(s)')
        plt.ylim((-4, 4))
        plt.title('observe dz')
