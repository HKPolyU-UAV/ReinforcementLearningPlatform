This folder is the definition of the robust RL for uav environment. This document is the detailed description of each file.

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

### Color
The various color definition for visualization.

### ref_cmd.py
The functions in this file are set to generate reference signals and disturbances of the UAV.

### uav.py
The mathematical model of the UAV.

### uav_att_ctrl.py
The class for UAV attitude control.

### uav_pos_ctrl.py
The class for UAV position control.

### uav_hover.py
The environment for UAV hover control using both inner-loop and outer-loop actions from RL.

### uav_hover_outer_loop.py
The environment for UAV hover control using virtual acceleration ux,uy,uz as actions from RL,
and FNTSMC for attitude control.

### uav_inner_loop.py
The environment for UAV attitude tracking control using torques Tx,Ty,Tz as actions from RL.

### uav_tracking_outer_loop.py
The environment for UAV position tracking control using virtual acceleration ux,uy,uz as actions from RL,
and FNTSMC for attitude control.