from map import Map

map = Map()
map.generate_circle_obs_training_dataset(xMax=5.0,
                                         yMax=5.0,
                                         safety_dis_obs=0.4,
                                         safety_dis_st=0.2,
                                         rMin=0.1,
                                         rMax=0.6,
                                         obsNum=3,
                                         batch=10,
                                         filename='dataset.txt')
