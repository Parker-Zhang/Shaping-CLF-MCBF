# Shaping-CLF-MCBF

This repository contains the code corresponding to the paper: 

**Deadlock-Aware Control for Multi-Robot Coordination with Multiple Safety Constraints**, in *IEEE Transactions on Robotics*, doi: 10.1109/TRO.2025.3600159.

The proposed framework focuses on detecting and avoiding deadlocks in multi-robot systems while ensuring safety and performance through the use of Control Lyapunov Functions (CLFs) and Control Barrier Functions (CBFs).

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
