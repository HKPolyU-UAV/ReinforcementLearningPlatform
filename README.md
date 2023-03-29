# Installation
This platform can be implemented on both windows and ubuntu since it is all established by Python. Currently, there is no ROS-related packages.
## Python version
Python 3.6 to 3.9 have been tested.
## Installation
Pre-installed: Anaconda3, any version that has a default python3.6 to 3.9 is fine.
```commandline
pip install opencv-python
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
The version of PyTorch depends on the device you have. You can choose CPU only or a specified CUDA version according to your GPU.
# ReinforcementLearning
Currently, this repository consists of algorithm, common, datasave, environment, and simulation.

Algorithm includes some commonly used reinforcement learning algorithms, DQN (Double, Dueling-), DDPG, TD3, for example.

Common includes common_func.py and common_cls.py containing some basic functions.
Common_func.py contains some commonly used functions, and common_cls.py contains some commonly defined classes.

Datasave saves networks trained by RL algorithms, and some data files.

Environment contains some physical models, which are called 'environment' in RL.
The 'config' directory contains the model description files for all environments, which are '**.xml' files.
THe 'envs' directory covers the ODE of the physical environments.

Simulation is the place where we implement our simulation experiments, which means, using different algorithms in different environments.

## Demos
Currently, we have DQN, DoubleDQN, DuelingDQN, DDPG, TD3, A2C, SAC, PPO, DPPO.

Currently, we have the following well-trained controllers

1. A DDPG controller for:

    FlightAttitudeSimulator,

    UGVBidirectional (trajectory planner),

    UGVForward (trajectory planner),

    UGVForwardObstacleAvoidance (trajectory planner).
2. A DQN controller for:

    FlightAttitudeSimulator. 
3. A TD3 trajectory planner for:

    UGVForwardObstacleAvoidance,

    CartPole,

    CartPoleAngleOnly,

    FlightAttitudeSimulator.
4. A PPO controller for:

    CartPoleAngleOnly,

    FlightAttitudeSimulator2State. 
5. A DPPO controller for:

    CartPoleAngleOnly,

    FlightAttitudeSimulator2State.

## Run the scripts
All runnable scripts are in './simulation/'.
### A DQN controller for a flight attitude simulator.
```commandline
cd simulation/DQN_based/
python3 DQN-4-Flight-Attitude-Simulator.py
```
The result should be similar to the following.

<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/dqn-4-flight-attitude-simulator.gif" width="400px">
</div>

In 'DQN-4-Flight-Attitude-Simulator.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
### A DDPG trajectory planner which can avoid obstacles for a forward-only UGV.
```commandline
cd simulation/PG_based/
python DDPG-4-UGV-Forward-Obstacle.py
```
The result should be similar to the following.

In 'DDPG-4-UGV-Forward-Obstacle.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Obstacle1.gif" width="400px"><img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Obstacle2.gif" width="400px">
</div>

### A DDPG trajectory planner in an empty world for a forward-only UGV.
```commandline
cd simulation/PG_based/
python DDPG-4-UGV-Forward.py
```
The result should be similar to the following.

In 'DDPG-4-UGV-Forward.py', set:
```commandline
    TRAIN = False
    RETRAIN = False
    TEST = not TRAIN
```
<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Forward1.gif" width="400px"><img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning/blob/main/datasave/video/gif/DDPG-4-UGV-Forward2.gif" width="400px">
</div>

# TODO
- [x] Add A2C.
- [ ] Add A3C.
- [x] Add PPO.
- [x] Add DPPO.
- [ ] Add DPPO2.
- [x] Train controllers for CartPole.
- [x] Add some PPO demos.
- [x] Add some DPPO demos.
- [ ] Add some A3C demos.
- [ ] Add more environments.