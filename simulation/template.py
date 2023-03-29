import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
'''
    This command is to add a path to this project. The environments can be loaded through PyCharm.
    But they cannot be loaded directly in a python terminal if we do not add the path of the files into the system environment variables.
'''

'''This is a template py file to show us how to establish our own training and testing project'''

cfgPath = '../../environment/config/'  # the config path, all model description files (.xml) are stored in this directory.
cfgFile = 'Two_Wheel_UGV.xml'  # the environment that you want to use
optPath = '../../datasave/network/'  # trained nets that can be tested
logPath = '../../datasave/log/'  # path to save the log file
show_per = 1

'''
    0. Re-write two classes: CriticNetWork and ActorNetWork. This is designed for simplicity since different training projects 
    have different net structures and parameters. The reference of CriticNetWork and ActorNetWork can be found in common.py.
'''

'''
    1. Two predefined functions. 
        fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool)
            Full fill the replay memory with current optimal neural network.
            """
            :param randomEnv:           init env randomly or not
            :param fullFillRatio:       fill ratio
            :param is_only_success:     only add successful episodes into the memory 
            :return:
            """
        fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float, is_only_success: bool)
            Full fill the replay memory randomly.
            """
            :param randomEnv:           init env randomly or not
            :param fullFillRatio:       fill ratio
            :param is_only_success:     only add successful episodes into the memory 
            :return:
            """
'''

if __name__ == '__main__':
    simulationPath = logPath + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + 'yourStr'
    '''
        'yourStr' consists of two parts: name of the algorithm and name of the env. Such as:
            'DDPG-Flight-Attitude-Simulator', 'PPO-Car', or 'DQG-UAV', etc.
    '''
    os.mkdir(simulationPath)  # create a new directory for this time training

    '''
        2. Define environment and agent (RL algorithm). 
            env = yourEnv()
            agent = RLAlgorithm()
    '''

    c = cv.waitKey(1)
    TRAIN = False  # train directly
    RETRAIN = False  # retrain
    TEST = not TRAIN  # test
    is_storage_only_success = True  # the same as 'is_only_success'
    '''
        Here, we define three working mode for the code.
            a. Train    b. Retrain    c. Test
    '''

    if TRAIN:
        successCounter = 0  # counter for successful episodes
        timeOutCounter = 0  # counter for timeout episodes
        failCounter = 0  # counter for failure episodes
        MAX_EPISODE = 5000  # maximum episode for this time training
        episode = 0

        if RETRAIN:
            print('...Retraining...')
            '''
                3-1: Function 'fullFillReplayMemory_with_Optimal' should be executed here.
                If the formation of the reward function before retraining and after retraining is different, then:
                    Neural Network must be re-initialized.
                Otherwise:
                    Neural Network reinitialization is optional.
            '''
        else:
            '''
                3-2: Function 'fullFillReplayMemory_Random' should be executed here.
            '''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        while episode <= MAX_EPISODE:
            # env.reset()
            print('=========START=========')
            print('Episode:', episode)
            '''
                reset the environment deterministicly or randomly.
                env.reset() or env.reset_random()
            '''
            sumr = 0  # cumulative reward
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()

            '''
                4. For each episode:
                    4.1 current state = next state.copy()
                    4.2 set epsilon
                    4.3 select action
                    4.4 env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                    4.5 graphic show
                    4.6 if is_storage_only_success:
                            new_state.append(env.current_state)
                            new_action.append(env.current_action)
                            new_reward.append(env.reward)
                            new_state_.append(env.next_state)
                            new_done.append(1.0 if env.is_terminal else 0.0)
                        else
                            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                5. if is_storage_only_success:  # 只用超时的或者成功的训练
                        print('Update Replay Memory......')
                        agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                6. data save
            '''  # Details for this process can be found in demos in 'simulation' directory

    if TEST:
        print('TESTing...')
        '''
            3.3 agent.load_actor_optimal(path=optPath, file='yourFile')
        '''

        '''
            The followings are user-defined module. You can test your network in any way you like.
        '''
