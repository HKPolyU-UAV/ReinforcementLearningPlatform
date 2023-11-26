from UGVForwardObstacleAvoidance import UGVForwardObstacleAvoidance as env
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    env = env()
    video = cv.VideoWriter(env.name + '.mp4', cv.VideoWriter_fourcc(*"mp4v"), 60, (env.image_size[0], env.image_size[1]))
    n = 5
    for i in range(n):
        env.reset(True)
        test_r = 0.
        while not env.is_terminal:
            a = np.random.uniform(env.action_range[:, 0], env.action_range[:, 1])
            env.step_update(a)
            test_r += env.reward
            env.visualization()
            video.write(env.image)
            # cv.waitKey(0)
        print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
    video.release()
