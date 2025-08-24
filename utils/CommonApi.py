"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         CommonApi.py
Description:    Functions commonly used for simulation.

Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
import numpy as np

def plot_circle(ax, pos,radius,circle_param):
    theta = np.linspace(0, 2*np.pi, 50)
    xc = pos[0]
    yc = pos[1]
    x = xc + radius*np.cos(theta)
    y = yc + radius*np.sin(theta)
    
    face_color = circle_param["face_color"]
    edge_color = circle_param["edge_color"]
    zorder = circle_param["zorder"]
    circle_type = circle_param["type"]
    alpha = circle_param.get("alpha", 1.0)
    
    if circle_type == "agent":
        ax.fill(x, y, facecolor = face_color, zorder = zorder, alpha=alpha)
        ax.plot(x, y, color = edge_color, linewidth = 1.0, zorder = zorder, linestyle = "-", alpha=alpha)
    if circle_type == "pos_mark":
        ax.plot(x, y, color = edge_color, linewidth = 1.5, zorder = zorder, linestyle = "--")

def inComAgent(index,agent_pos,Cr):
    index_ = []
    agentN = np.shape(agent_pos)[0]
    for i in range(agentN):
        if i != index:
            if np.linalg.norm(agent_pos[index,0:2]-agent_pos[i,0:2]) <= Cr:
                index_.append(i)
    return index_


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return None
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return [r, g, b]

def compute_traj_len(traj):
    step, dim = traj.shape
    traj_len = 0
    for k in range(step-1):
        len_ = np.linalg.norm(traj[k+1,:2]-traj[k,:2])
        traj_len += len_
    return traj_len
