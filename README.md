# Introduction

This is a **Reinforcement Learning Control** simulation platform that integrates many different **Algorithms** and **Environment** together.
It does not require some complicated python packages, for example, gym, mujoco, casadi, acado, et al.
This platform only requires pytorch, opencv-python, numpy, matplotlib, et al., all of which are easy to install and config.
The objective is to allow developers to create their own environments and algorithms totally by themselves and would not blocked by some other python packages.
Recently, we have updated this platform.

# Modification

**In a nutshell! We rewrite the whole platform!**

* All environments have been rewritten. We remove model description files from environments, which are useless.
  We further standardized some basic functions and parameters to enhance it portability.
* All algorithms have been rewritten. We remove many redundant modules in our previous version. For example, network saving functions, network loading functions,
  and some variables defined but never used.
* All basic classes have been rewritten. We remove some useless classes in /utils/classes.
* All demonstrations have been re-trained. The demos we have already trained are not standardized. Therefore, we retrain all of them.

# Installation

Very easy, you even cannot use "install" to define this operation. You can firstly install nothing, just download this platform, run it, and then see
what packages you need to install. Mainly, pytorch (gpu version, cpu is also ok for now), opencv-python, numpy, pandas, matplotlib.

# Reinforcement Learning Platform

This platform currently consists of four parts, namely, **algorithm**, **demonstration**, **environment**, and **utils**.

## utils

Nothing special, just some commonly used classes and functions.

## algorithm

All RL algorithms we implemented for now.

|  Algorithm  |  classification |         Description          |
| ----------- | --------------- | ---------------------------- |
|     DQN     |   value based   |              None            |
| Dueling DQN |   value based   |              None            |
| Double DQN  |   value based   |              None            |
|     DDPG    |   actor-critic  |              None            |
|     TD3     |   actor-critic  |        update of DDPG        |
|     SAC     |   actor-critic  |              None            |
|     PPO     |   policy based  |              None            |
|     DPPO    |   policy based  |      multi process PPO       |
|     PPO2    |   policy based  | PPO with gae and some tricks |
|    DPPO2    |   policy based  |      multi process PPO2      |

## environment

### 1. BallBalancer

A ball is put on a beam, and the beam is controlled by a two-link manipulator. The objective is to make the ball hold at the center of the beam by adjusting the joints' angle of the of manipulator.
![image](https://github.com/ZhengYaWei1992/ZWProgressView/blob/master/Untitled3.gif)

### 2. CartPole

A cartpole, nothing special. Two environments are integrated. The first is angluar only, the objectove is to balance the pole by adjusting the force added on the cart. The objective of the second is to control both the angle of the cart and the position of the pole.

### 3. FlightAttitudeSimulator

This is an intermediate fixed rod with free rotational capability. We need to keep the rod in a horizontal position by adjusting the force added at the end of the rod.

### 4. RobotManipulator

A two-link manipulator. The objective is to control the position of the end of the manipulator.

### 5. SecondOrderIntegration

A mass point, the control input is the two-dimensional accelection, we need to control its position.

### 6. UavFntsmcParam

A quadrotor has already controlled by fast nonsingular terminal sliding mode controller (FNTSMC). We use RL to automatically tune the hyper-parameters of the FNTSMC to achieve a better tracking performance. Also, we have position tracking mode and attitude tracking mode.

### 7. UavRobust

Just quadrotor fixed-point control. The difference between **UavFntsmcParam** and **UavRobust** is that **UavRobust** directly use RL to control the quadrotor while **UavFntsmcParam** utilizes RL to optimize the hyper-parameters of FNTSMC.

### 8. UGV

A ground vehicle, the control outputs are the expected linear and angular accelections. The objective is to control the position of the UGV.





