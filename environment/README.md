This directory storages some environments for us to do some simulation experiments.

Currently, there are two sub-directories: config and env.

# Config
This directory storages all model description files in it. A model description file is an 'xml' file that generated 
automatically by the corresponding environment.

Currently, there are only two models in it: 'Flight_Attitude_Simulator.xml' and 'Flight_Attitude_Simulator_Continuous.xml'.
Actually, they are identical except the action space. The action of the former is discrete while the latter is continuous. Apparently, the
former one is designed for value-based RL, while the latter one is designed for policy-based RL.

# Envs
This directory contains source files of the models. However, it contains more files than that of in 'config' because some environments are defined and 
described by more than one python files.

To figure out which python file is the 'exact' source file of the 'xml', we can check the name of the file, which should be identical to the corresponding
'xml' file in 'config'.

# Environments we have:
## Flight attitude simulator:
It is a single degree of freedom flight attitude simulator. The input ot the system is the force at the right end
of the rod, and the output is the degree of the rod. We can design a RL controller for the system.

## Two wheel ground vehicle:
It is the model of a two-wheel differential ground vehicle. However, this model is not for controlling
problem but for planning problem. The entire model is the differential ground vehicle AND an empty world with
4mx4m.  We can use this environment to learn how to train an agent for a trajectory planning problem.

Currently, these are the models we have.