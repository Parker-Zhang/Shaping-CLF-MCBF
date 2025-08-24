#%% -*- coding: utf-8 -*-
"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         sim-random_obs_navigation.py
Description:    
This script implements multi-robot navigation in two scenarios:
1. Robots exchange their positions in environments with random obstacles.
2. Robots move to random target locations in environments with random obstacles.
Note: No path planner is used; only a reactive controller is implemented. 
In real applications, path planning can be integrated to improve motion efficiency.
Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
import matplotlib.pyplot as plt
import numpy as np
from agents.circle2DAgent import Circle2DAgent
from workspace.workspace import WorkSpace
import copy
import time
from utils.CommonApi import *

test_N = 1 # test number
fig_flag = True
use_env_seed_list = False
env_seed_list = [300, ]
if use_env_seed_list:
    test_N = len(env_seed_list)

class Cfg:
    # task config
    agentN = 20
    obs_N =  25
    targetM = copy.deepcopy(agentN)
    test_seed = 50  # random seed
    agent_pos_init_style = 'circle_exchange_pos' # random_init_tar_pos, circle_exchange_pos
    circle_radius = 6

    # robot Cfg
    sense_radius = 1.0
    com_Rc = 1.0
    max_vel = 1.0
    agent_r = 0.2

    # Env Cfg
    env_seed = 100
    map_corner_pos = [[-7,-7], [7,7]]
    obs_map_corner_pos = [[-4.5,-4.5], [4.5,4.5]]
    obs_flag = True
    obs_radius = 0.4
    obs_inter_dist = 0.3
    obs_type = 'circ'
    obs_pos_list = []

    # simulation Cfg
    max_step = 1000
    dt = 0.1
    sim_stop_err = 0.05

    # controller Cfg
    cbf_boundary_hold = {"agentCA":0.1, "obsCA":0.1}
    nei_vel_flag = True
    CLF_shaping_flag = True
    clf_alpha = 3
    clf_rho = 0.5

#%% 
np.random.seed(Cfg.test_seed)
agent_color = np.array([0,0,255])/255
agent_face_color = np.array([99,128,253])/255

deadlock_face_color = ['#00aa8c', '#6fb3ee', '#ffe143', '#e37158']
deadlock_face_color = [hex_to_rgb(color) for color in deadlock_face_color]
deadlock_edge_color = ['#129400', '#0000FF', '#FFA006', '#E92E2B']
deadlock_edge_color = [hex_to_rgb(color) for color in deadlock_edge_color]

map_corner_pos = np.array(Cfg.map_corner_pos)
obs_map_corner_pos = np.array(Cfg.obs_map_corner_pos)

traj_len_save = []
swarm_traj_len_all = []

#%% start tests
for k in range(test_N):
    print('----- start {}th test!'.format(k+1))
    env_seed = np.random.randint(1,10000)
    if use_env_seed_list:
        env_seed = env_seed_list[k]
    np.random.seed(env_seed)
    print('env_seed:', env_seed)
    Cfg.env_seed = env_seed
    ws = WorkSpace(obs_map_corner_pos)
    # init env
    if Cfg.obs_flag:
        # random generate disjoint obstacles
        obs_info = []
        obs_pos_list = []
        obs_shape_param = {'type':Cfg.obs_type, 'charact':{'radius':Cfg.obs_radius + Cfg.obs_inter_dist}}
        obs_pos_set = ws.radomGeneratePosFromDict(Cfg.obs_N, obs_shape_param) # [array]
        print('obs_pos_set num', obs_pos_set.shape[0])
        for i in range(obs_pos_set.shape[0]):
            obs_name = 'unsafeR'+str(i)
            obs_pos = obs_pos_set[i].tolist()
            obs_pos_list.append(obs_pos)
            obs_charact = {'cen_pos':obs_pos, 'radius':Cfg.obs_radius}
            obs = {'name':obs_name,'type':Cfg.obs_type,'charact':obs_charact}
            obs_info.append(obs)
        Cfg.obs_pos_list = obs_pos_list
        ws.add_obs_from_dict(obs_info)
    print('Cfg.obs_pos_list: ', Cfg.obs_pos_list)
    ws.change_map_size(map_corner_pos)

    # init agent position and target position
    agent_init_pos = []
    agent_tar_pos_set = []
    if Cfg.agent_pos_init_style == 'circle_exchange_pos':
        Cfg.nei_vel_flag = False
        Cfg.cbf_boundary_hold = {"agentCA":0.1, "obsCA":0.1}
        for i in range(Cfg.agentN):
            angle = 2*np.pi/Cfg.agentN*i
            agent_init_pos.append([round(Cfg.circle_radius*np.cos(angle),2), round(Cfg.circle_radius*np.sin(angle),2)])
            agent_tar_pos_set.append([round(Cfg.circle_radius*np.cos(angle+np.pi),2), round(Cfg.circle_radius*np.sin(angle+np.pi),2)])
    elif Cfg.agent_pos_init_style == 'random_init_tar_pos':
        Cfg.nei_vel_flag = True
        Cfg.cbf_boundary_hold = {"agentCA":0.1, "obsCA":0.1}
        agent_shape_param = {"type":"circ", "charact": {'radius':Cfg.agent_r+Cfg.obs_inter_dist}}
        agent_init_pos = ws.radomGeneratePosFromDict(Cfg.agentN, agent_shape_param)
        agent_tar_pos_set = ws.radomGeneratePosFromDict(Cfg.agentN, agent_shape_param)
    agent_init_pos = np.array(agent_init_pos)
    agent_tar_pos_set = np.array(agent_tar_pos_set)

    #% init robots
    agent_set = []
    for i in range(Cfg.agentN):
        agent_set.append(Circle2DAgent())
        agent_set[i].cur_pos = agent_init_pos[i]
        agent_set[i].cbf_boundary_hold = Cfg.cbf_boundary_hold
        agent_set[i].sense_radius = Cfg.sense_radius
        agent_set[i].init_workspace(ws)
        agent_set[i].CLF_shaping_flag = Cfg.CLF_shaping_flag
        agent_set[i].alpha = Cfg.clf_alpha
        agent_set[i].rho = Cfg.clf_rho
        agent_set[i].nei_vel_flag = Cfg.nei_vel_flag
        agent_set[i].agent_r = Cfg.agent_r
        agent_set[i].v_max = Cfg.max_vel
        agent_set[i].tar_pos = agent_tar_pos_set[i]

    if fig_flag:
        fig = plt.figure()
        ax = fig.subplots()
        ws.plt_workspace(ax)
        for i in range(Cfg.agentN):
            agent_face_color  = deadlock_face_color[agent_set[i].deadlock_type-1]
            agent_color = deadlock_edge_color[agent_set[i].deadlock_type-1]
            circle_param = {"edge_color":agent_color, "face_color":agent_face_color, 'zorder':5, "type":'agent'}
            plot_circle(ax, agent_set[i].cur_pos,agent_set[i].agent_r, circle_param)
            ax.plot([agent_set[i].cur_pos[0], agent_set[i].tar_pos[0]], [agent_set[i].cur_pos[1], agent_set[i].tar_pos[1]], color = agent_color, linewidth = 1.5, linestyle = "--")

    # record data
    pos_save = []
    vel_save = []
    replan_time = []
    finish_step_save = []

    for i in range(Cfg.agentN):
        pos_save.append(np.empty((0,2)))
        vel_save.append(np.empty((0,2)))
        replan_time.append([])
        finish_step_save.append(np.inf)

    np.set_printoptions(precision=3)
    step = 0
    finish_step = np.inf

    # simulation loop
    while step < Cfg.max_step:
        swarm_pos = np.array([agent_set[i].cur_pos.tolist() for i in range(Cfg.agentN)])
        for i in range(Cfg.agentN):
            error = np.linalg.norm(agent_set[i].cur_pos - agent_tar_pos_set[i])
            if error < Cfg.sim_stop_err and finish_step_save[i] == np.inf:
                finish_step_save[i] = step+1

        # is task finished?
        error = [np.linalg.norm(agent_set[i].cur_pos - agent_tar_pos_set[i]) 
                for i in range(Cfg.agentN)]
        if max(error) < Cfg.sim_stop_err:
            finish_step = step
            break

        # sensor data
        for i in range(Cfg.agentN):
            nei_index = inComAgent(i,swarm_pos,Cfg.com_Rc)
            nei_pos = [agent_set[index].cur_pos for index in nei_index]
            nei_vel = [agent_set[index].vel for index in nei_index]
            nei_deadlock_type = [agent_set[index].deadlock_type for index in nei_index]
            agent_set[i].nei_agent_pos = copy.deepcopy(nei_pos)
            agent_set[i].nei_agent_vel = copy.deepcopy(nei_vel)
            agent_set[i].nei_agent_deadlock_type = copy.deepcopy(nei_deadlock_type)

        # run controller
        for i in range(Cfg.agentN):
            start=time.time()
            agent_set[i].CLF_CBF_QP_controller()
            time_interval=time.time()-start
            replan_time[i].append(time_interval)

        # record data
        for i in range(Cfg.agentN):
            pos_save[i] = np.vstack([pos_save[i], agent_set[i].cur_pos])
            vel_save[i] = np.vstack([vel_save[i], agent_set[i].vel])

        # update robot state
        for i in range(Cfg.agentN):
            agent_set[i].update_state(Cfg.dt)

        if fig_flag:
            ax.clear()
            ws.plt_workspace(ax)
            for i in range(Cfg.agentN):
                agent_face_color  = deadlock_face_color[agent_set[i].deadlock_type-1]
                agent_color = deadlock_edge_color[agent_set[i].deadlock_type-1]
                circle_param = {"edge_color":agent_color, "face_color":agent_face_color, 'zorder':5, "type":'agent'}
                plot_circle(ax, agent_set[i].cur_pos,agent_set[i].agent_r, circle_param)
                circle_param = {"edge_color":agent_color, "face_color":agent_face_color, 'zorder':1, "type":'pos_mark'}
                plot_circle(ax, agent_set[i].tar_pos,agent_set[i].agent_r/2, circle_param)
                ax.plot([agent_set[i].cur_pos[0], agent_set[i].tar_pos[0]], [agent_set[i].cur_pos[1], agent_set[i].tar_pos[1]], color = agent_color, linewidth = 1.5, linestyle = "--")
            plt.pause(0.01)
        step += 1

    # performance summary
    print('finish task in {} seconds'.format(round(step*Cfg.dt,2)))
    pos_save_list = [agent_pos_.tolist() for agent_pos_ in pos_save]
    vel_save_list = [agent_vel_.tolist() for agent_vel_ in vel_save]
    # traj len
    swarm_traj_len = [round(compute_traj_len(pos_save[i]),2) for i in range(Cfg.agentN)] # 总路径长度
    traj_len_save.append(sum(swarm_traj_len))
    print('average traj len:', sum(swarm_traj_len)/Cfg.agentN)
    swarm_traj_len_all.append(swarm_traj_len)
    # replan time
    replan_time_mean = [np.mean(replan_time[i]) for i in range(Cfg.agentN)]
    print('average replan time:', np.mean(replan_time_mean))