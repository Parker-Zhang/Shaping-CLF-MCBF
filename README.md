# Shaping-CLF-MCBF

This repository contains the code corresponding to the paper titled: **Deadlock-Aware Control for Multi-Robot Coordination with Multiple Safety Constraints**. The proposed framework focuses on detecting and avoiding deadlocks in multi-robot systems while ensuring safety and performance through the use of Control Lyapunov Functions (CLFs) and Control Barrier Functions (CBFs).

**Note**: This repository currently contains an example of robots encountering each other in a narrow passage; the complete code will be made publicly available upon the paper’s publication.

## Features
- **Distributed Control Framework**
- **Multiple Safety Constraints**
- **Potential Deadlock Detection**
- **Reactive Deadlock Avoidance**
- **Comparison with Other Methods**
  - **LSC**: Linear Safety Corridor [Github](https://github.com/qwerty35/lsc_dr_planner)
  - **RSC**: Recursive Safety Corridor [Github](https://github.com/PKU-MACDLab/IMPC-OB)
  - **CBF-CD**: CBF with Consistent Disturbance [paper](https://ieeexplore.ieee.org/document/7857061)

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`
- `qpsolvers`
