#%% -*- coding: utf-8 -*-
"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         sim-narrow_passage.py
Description:    Simulations of robots encounter in a narrow passage.
                - sim1: Position swap of two robots in a narrow passage.
                - sim2: Four robots encounter in a narrow passage.
                - sim3: Position swap of two robots in a narrow passage with target offset.
                - sim4: Four robots encounter in a narrow passage with target offset.
                - sim5: Position swap of two robots in a wider passage.
                  
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
from utils.CommonApi import *

fig_flag = True
result_fig_flag = False

class Cfg:
    agentN = 2  # or 4
    targetM = copy.deepcopy(agentN)
    sense_radius = np.inf
    com_Rc = np.inf
    max_vel = 1.0
    agent_r = 0.2
    total_time = 80  # simulation time
    dt = 0.1
    max_step = int(total_time/dt)+1
    sim_stop_err = 0.05
    map_corner_pos = [[-3.5,-1.5], [3.5,1.5]]
    
    # CBF setup
    cbf_boundary_hold = {"agentCA":0.5, "obsCA":0.01}
    # cbf_boundary_hold = {"agentCA":0.5, "obsCA":0.2}
    CLF_shaping_flag = True
    angle_hold = 10  # dist to deadlock, deg to dist
    

#%%
deadlock_face_color = ['#00aa8c', '#6fb3ee', '#ffe143', '#e37158']
deadlock_face_color = [hex_to_rgb(color) for color in deadlock_face_color]
deadlock_edge_color = ['#129400', '#0000FF', '#FFA006', '#E92E2B']
deadlock_edge_color = [hex_to_rgb(color) for color in deadlock_edge_color]
map_corner_pos = np.array(Cfg.map_corner_pos)
ws = WorkSpace(map_corner_pos)

# init workspace
obs_info = []
# ---- wider passage
# obs_wall_up = {'name':'up_wall', 'type': 'rect', 'charact': {'left_corner': [-2,0.3], 'right_corner': [2,0.4]}}
# obs_wall_down = {'name':'down_wall', 'type': 'rect', 'charact': {'left_corner': [-2,-0.4], 'right_corner': [2,-0.3]}}

obs_wall_up = {'name':'up_wall', 'type': 'rect', 'charact': {'left_corner': [-2,0.2], 'right_corner': [2,0.4]}}
obs_wall_down = {'name':'down_wall', 'type': 'rect', 'charact': {'left_corner': [-2,-0.4], 'right_corner': [2,-0.2]}}

obs_info.append(obs_wall_up)
obs_info.append(obs_wall_down)

# obs_wall_left = {'name':'left_wall', 'type': 'rect', 'charact': {'left_corner': [-2,-0.2], 'right_corner': [-1.8,0.2]}}
# obs_wall_right = {'name':'right_wall', 'type': 'rect', 'charact': {'left_corner': [1.8,-0.2], 'right_corner': [2,0.2]}}
# obs_info.append(obs_wall_left)
# obs_info.append(obs_wall_right)

ws.add_obs_from_dict(obs_info)

if Cfg.agentN == 4:
    Cfg.angle_hold = 20
    agent_color_set = [np.array([68,114,196])/255, np.array([68,114,196])/255,
                       np.array([255,192,0])/255, np.array([255,192,0])/255]
    
    agent_init_pos = np.array([[-0.6,0],[-0.2,0],[0.2,0],[0.6,0]])

    # --- sim 2
    # agent_tar_pos_set = np.array([[1.6,0],[1.2,0],[-1.2,0],[-1.6,0]])

    # --- sim 4
    agent_tar_pos_set = np.array([[2.8,0.2],[2.4,0.2],[-2.4,-0.2],[-2.8,-0.2]])

else:
    Cfg.angle_hold = 10
    Cfg.agentN = 2
    agent_color_set = [np.array([68,114,196])/255, np.array([255,192,0])/255]
    # --- sim 1 and 5
    # agent_init_pos = np.array([[-1,0],[1,0]])
    # agent_tar_pos_set = np.array([[1,0],[-1,0]])

    # --- sim 3
    agent_init_pos = np.array([[-1,0],[1,0]])
    agent_tar_pos_set = np.array([[2.5,0.2],[-2.5,-0.2]])

    # --- sim 6: successful parameters for position swap in a wider passage
    # Cfg.angle_hold = 30
    # Cfg.cbf_boundary_hold = {"agentCA":0.5, "obsCA":0.2}
    # agent_tar_pos_set = np.array([[2.8,0.],[-2.8,-0.]])

#% init agents
agent_set = []
for i in range(Cfg.agentN):
    agent_set.append(Circle2DAgent())
    agent_set[i].cur_pos = agent_init_pos[i]
    agent_set[i].cbf_boundary_hold = Cfg.cbf_boundary_hold
    agent_set[i].sense_radius = Cfg.sense_radius
    agent_set[i].init_workspace(ws)
    agent_set[i].CLF_shaping_flag = Cfg.CLF_shaping_flag
    agent_set[i].agent_r = Cfg.agent_r
    agent_set[i].v_max = Cfg.max_vel
    agent_set[i].deadlock_solver.angle_hold = Cfg.angle_hold
    agent_set[i].tar_pos = agent_tar_pos_set[i]

if fig_flag:
    fig = plt.figure()
    ax = fig.subplots()
    ws.plt_workspace(ax)
    for i in range(Cfg.agentN):
        agent_face_color  = deadlock_face_color[agent_set[i].deadlock_type-1]
        agent_edge_color = deadlock_edge_color[agent_set[i].deadlock_type-1]
        # agent_face_color = agent_color_set[i % len(agent_color_set)]
        # agent_edge_color = 'black'

        circle_param = {"edge_color":agent_edge_color, "face_color":agent_face_color, 'zorder':5, "type":'agent'}
        plot_circle(ax, agent_set[i].cur_pos,agent_set[i].agent_r, circle_param)
    # plt.show()

np.set_printoptions(precision=3)
step = 0
finish_step = np.inf
min_inter_dist = np.inf
collision_flag = False

# record data
t_save = []
pos_save = []
vel_save = []
deadlock_save = []
finish_step_save = []
ctl_res_save = []

for i in range(Cfg.agentN):
    pos_save.append(np.empty((0,2)))
    vel_save.append(np.empty((0,2)))
    deadlock_save.append([])
    finish_step_save.append(np.inf)
    ctl_res_save.append([])

#%% start simulation
while step < Cfg.max_step:
    # Determine whether to end the simulation
    swarm_pos = np.array([agent_set[i].cur_pos.tolist() for i in range(Cfg.agentN)])
    for i in range(Cfg.agentN):
        error = np.linalg.norm(agent_set[i].cur_pos - agent_tar_pos_set[i])
        if error < Cfg.sim_stop_err and finish_step_save[i] == np.inf:
            finish_step_save[i] = step+1
    error = [np.linalg.norm(agent_set[i].cur_pos - agent_tar_pos_set[i]) 
            for i in range(Cfg.agentN)]
    if max(error) < Cfg.sim_stop_err:
        finish_step = step
        break

    # Sensing
    for i in range(Cfg.agentN):
        nei_index = inComAgent(i,swarm_pos,Cfg.com_Rc)
        nei_pos = [agent_set[index].cur_pos for index in nei_index]
        nei_vel = [agent_set[index].vel for index in nei_index]
        agent_set[i].nei_agent_pos = copy.deepcopy(nei_pos)
        agent_set[i].nei_agent_vel = copy.deepcopy(nei_vel)

    # Contorl
    ctl_res = []
    for i in range(Cfg.agentN):
        res = agent_set[i].CLF_CBF_QP_controller()
        ctl_res.append(res)
        ctl_res_save[i].append(res)

    # record data
    t_save.append(step*Cfg.dt)
    for i in range(Cfg.agentN):
        pos_save[i] = np.vstack([pos_save[i], agent_set[i].cur_pos])
        vel_save[i] = np.vstack([vel_save[i], agent_set[i].vel])
        deadlock_save[i].append(agent_set[i].deadlock_type)

    # update agents' states
    for i in range(Cfg.agentN):
        agent_set[i].update_state(Cfg.dt)

    # plot
    if fig_flag:
        ax.clear()
        ws.plt_workspace(ax)
        for i in range(Cfg.agentN):
            agent_face_color  = deadlock_face_color[agent_set[i].deadlock_type-1]
            agent_edge_color = deadlock_edge_color[agent_set[i].deadlock_type-1]
            # agent_face_color = agent_color_set[i % len(agent_color_set)]
            # agent_edge_color = 'black'
            circle_param = {"edge_color":agent_edge_color, "face_color":agent_face_color, 'zorder':5, "type":'agent'}
            plot_circle(ax, agent_set[i].cur_pos,agent_set[i].agent_r, circle_param)
            # circle_param = {"edge_color":agent_face_color, "face_color":agent_face_color, 'zorder':1, "type":'pos_mark'}
            # plot_circle(ax, agent_set[i].tar_pos,agent_set[i].agent_r/2, circle_param)
            ax.scatter(agent_set[i].tar_pos[0], agent_set[i].tar_pos[1],
                       color=agent_face_color, marker='x',s=50, label='target', zorder=6)
            agent_set[i].plt_key_vector(ax, ctl_res[i])
        ax.set_title('step: {}, time: {:.2f}s'.format(step, step*Cfg.dt))
        plt.pause(0.01)
    step += 1


#%% plot agents' trajectories
label_flag = False
font_size = 16
line_width = 2.5
fig = plt.figure(figsize=(6, 3))
ax = fig.subplots()
larger_corner = np.array([[-3,-1.5], [3,1.5]])
ws.change_map_size(np.array(larger_corner))
# ws.change_map_size(np.array(Cfg.map_corner_pos))
ws.plt_workspace(ax)
for i in range(Cfg.agentN):
    agent_face_color = agent_color_set[i % len(agent_color_set)]
    traj = np.array(pos_save[i])
    ax.plot(traj[:-1,0], traj[:-1,1], color = 'blue', linewidth = line_width, zorder = 10, linestyle = "-")
    traj_len = len(traj)
    last_traj_pos = np.array([-1.5,1.5])
    for k in range(traj_len-1):
        if k % 2 == 0:
            alpha = 0.2 + (k / (traj_len - 1)) * 0.1
            traj_pos = traj[k]
            if np.linalg.norm(traj_pos - last_traj_pos) < 0.03 and k < traj_len-20:
                continue
            circle_param = {"edge_color":'black', "face_color":agent_face_color, 'zorder':5, "type":'agent', 'alpha': alpha}
            plot_circle(ax, traj_pos, Cfg.agent_r, circle_param)
            last_traj_pos = traj_pos

k = -1
for i in range(Cfg.agentN):
    agent_face_color = agent_color_set[i % len(agent_color_set)]
    traj = np.array(pos_save[i])
    ax.plot(traj[:-1,0], traj[:-1,1], color = 'blue', linewidth = line_width, zorder = 10, linestyle = "-")
    traj_len = len(traj)
    last_traj_pos = np.array([-1.5,1.5])
    traj_pos = traj[k]
    if np.linalg.norm(traj_pos - last_traj_pos) < 0.03 and k < traj_len-20:
        continue
    circle_param = {"edge_color":'black', "face_color":agent_face_color, 'zorder':5, "type":'agent', 'alpha': 1}
    plot_circle(ax, traj_pos, Cfg.agent_r, circle_param)
    last_traj_pos = traj_pos

ax.tick_params(axis='both', direction='in',length=5, width = 1.5, right = True, top = True)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.show()
