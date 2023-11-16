This folder is the definition of the Rl environment. This document is the detailed description of each file.

### collector
The class for data collection.

### FNTSMC
The file is designed for UAV control, which contains three classes, namely:
* fntsmc_param:
    the class that sets the hyper-parameters for FNTSMC controller
* fntsmc_pos:
    position controller
* fntsmc_att:
    attitude controller

### ref_cmd.py
The functions in this file are set to generate reference signals and disturbances of the UAV.

### uav.py
The mathematical model of the UAV.

### uav_att_ctrl.py
The class for UAV attitude control.

### uav_pos_ctrl.py
The class for UAV position control.
