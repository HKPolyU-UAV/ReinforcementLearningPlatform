# Installation
This platform can be implemented on both windows and ubuntu if you have installed python3. Currently, there is no ROS-related packages.
## Python version
Python higher than 3.8 is recommended.
## Installation
Pre-installed: Anaconda3, any version that has a default python higher than 3.8 is fine.
```commandline
pip install opencv-python
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
The version of PyTorch depends on the device you have. You can choose CPU only or a specified CUDA version according to your GPU.

# ReinforcementLearning
Currently, this repository consists of algorithm, common, datasave, environment, and simulation five parts..

**Algorithm**

    Algorithm includes some commonly used reinforcement learning algorithms.
    The following table lists RL algorithms in the corresponding directories.


| **Directory** |          **Algorithm**           |           Description           |
|:-------------:|:--------------------------------:|:-------------------------------:|
| actor_critic  |   A2C<br/>DDPG<br/>SAC<br/>TD3   |              ----               |
|  policy_base  |      PPO<br/>DPPO<br/>DPPO2      | ----<br/>----<br/>does not work |
|  value_base   | DQN<br/>DoubleDQN<br/>DuelingDQN |          not debugged           |
|    rl_base    |               ----               |Basic class that inherited <br/>by other algorithm classes|



**Common** 

    Common includes common_func.py and common_cls.py containing some basic functions.
    The following table lists the contents of the two py files.
|      **File**      |                        Description                        |
|:------------------:|:---------------------------------------------------------:|
| **common_cls.py**  | ReplayBuffer, RolloutBuffer, OUNoise, NeuralNetworks, etc |
| **common_func.py** |  basic mathematical functions, geometry operations, etc   |

**Datasave** 

    Datasave saves networks trained by RL algorithms and some data files.

**Environment** 

    Environment contains some physical models, which are called 'environment' in RL.
    The 'config' directory contains the **.xml file, the model description files of all environments.
    The 'envs' directory covers the ODE of the physical environments.
    The following table lists all the current environments.

|             **Environment**             |       **Directory**        |                     **Description**                      |
|:---------------------------------------:|:--------------------------:|:--------------------------------------------------------:|
|                CartPole                 |        ./CartPole/         |              continuous, position and angle              |
|            CartPoleAngleOnly            |        ./CartPole/         |                  continuous, just angle                  |
|        CartPoleAngleOnlyDiscrete        |        ./CartPole/         |                   discrete, just angle                   |
|         FlightAttitudeSimulator         | ./FlightAttitudeSimulator/ |                         discrete                         |
| FlightAttitudeSimulator2StateContinuous | ./FlightAttitudeSimulator/ |       continuous, state are only theta and dtheta        |
|    FlightAttitudeSimulatorContinuous    | ./FlightAttitudeSimulator/ |                        continuous                        |
|                UAVHover                 |           ./UAV/           | continuous, other files in ./UAV are not RL environments |
|            UGVBidirectional             |           ./UGV/           |  continuous, the vehicle can move forward and backward   |
|               UGVForward                |           ./UGV/           |      continuous, the vehicle can only move forward       |
|           UGVForwardDiscrete            |           ./UGV/           |       discrete, the vehicle can only move forward        |
|      UGVForwardObstacleContinuous       |           ./UGV/           |     continuous, the vehicle needs to avoid obstacles     |
|       UGVForwardObstacleDiscrete        |           ./UGV/           |      discrete, the vehicle needs to avoid obstacles      |

**Simulation** 

    Simulation is the place where we implement our simulation experiments,
    which means, using different algorithms in different environments.

## Demos
Currently, we have the following well-trained controllers:

### **DDPG**

A DDPG controller for
* FlightAttitudeSimulator
* UGVBidirectional (motion planner)
* UGVForward (motion planner)
* UGVForwardObstacleAvoidance (motion planner)

### **DQN**

A DQN controller for
* FlightAttitudeSimulator

### **TD3**

A TD3 trajectory planner for:
* UGVForwardObstacleAvoidance
* CartPole
* CartPoleAngleOnly
* FlightAttitudeSimulator

### **PPO**

A PPO controller for:
* CartPoleAngleOnly
* FlightAttitudeSimulator2State 

### **DPPO**

A DPPO controller for:

* CartPoleAngleOnly
* FlightAttitudeSimulator2State

## Run the scripts
All runnable scripts are in './simulation/'.
### A DQN controller for a flight attitude simulator.

In 'DQN-4-Flight-Attitude-Simulator.py', set: (set TRAIN to be True if you want to train a new controller)
```commandline
 TRAIN = False
 RETRAIN = False
 TEST = not TRAIN
```

In command window:
```commandline
cd simulation/DQN_based/
python3 DQN-4-Flight-Attitude-Simulator.py
```
The result should be similar to the following.

<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning_V2/blob/main/datasave/video/gif/dqn-4-flight-attitude-simulator.gif" width="400px">
</div>


### A DDPG motion planner which can avoid obstacles for a forward-only UGV.
In 'DDPG-4-UGV-Forward-Obstacle.py', set: (set TRAIN to be True if you want to train a new motion planner)
```commandline
 TRAIN = False
 RETRAIN = False
 TEST = not TRAIN
```

In command window:
```commandline
cd simulation/PG_based/
python DDPG-4-UGV-Forward-Obstacle.py
```
The result should be similar to the following.

<div align=center>
    <img src="https://github.com/ReinforcementLearning-StudyNote/ReinforcementLearning_V2/blob/main/datasave/video/gif/DDPG-4-UGV-Obstacle2.gif" width="400px">
</div>

# TODO
## Algorithms
- [x] Add A2C
- [ ] Add A3C
- [x] Add PPO
- [x] Add DPPO
- [ ] Add D4PG

## Demo
- [x] Train controllers for CartPole
- [x] Add some PPO demos
- [x] Add some DPPO demos
- [ ] Add some A3C demos

## Environments
- [ ] Modify UGV (add acceleration loop)
- [ ] Add a UAV regulator
- [ ] Add a UAV tracker
- [ ] Add a 2nd-order integration system
- [ ] Add a duel-joint robotic arm
- [ ] Add a 2nd-order cartpole (optional)

## Debug
- [ ] Debug DPPO2
- [ ] Debug DQN-based algorithms (multi-action agents)