# -*- coding: utf-8 -*-
"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         workspace.py
Description:    Simulated workspace.

Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as Polygon_plt
from shapely.affinity import rotate, translate
import numpy as np
import copy
from rtree import index

class WorkSpace():
    def __init__(self, map_corner_pos):
        self.change_map_size(map_corner_pos)
        self.clear_obs()

    def change_map_size(self, map_corner_pos):
        self.left_corner_pos = map_corner_pos[0,:] 
        self.right_corner_pos = map_corner_pos[1,:] 
        self.map_corner_pos = copy.deepcopy(map_corner_pos) 
        self.map_range = map_corner_pos.T.flatten().tolist() 
        self.map_size = [self.map_range[1]-self.map_range[0],self.map_range[3]-self.map_range[2]] 

    def clear_obs(self):
        self.obs_poly_set = dict() 
        self.obs_circ_set = []
        self.dy_obs_circ_set = []
        self.rt_idx = index.Index()
        self.obs_index = 0
        self.rt_obs_set = []
        self.rt_obs_type = []

    def add_obs_from_dict(self, obs_info):
        for obs in obs_info:
            obs_name = obs['name']
            obs_type = obs['type']
            obs_charact = obs['charact']
            if obs_type == 'rect':
                left_pos = obs_charact['left_corner']
                right_pos = obs_charact['right_corner']
                length = right_pos[0] - left_pos[0]
                width = right_pos[1] - left_pos[1]
                poly = Polygon([(left_pos[0],left_pos[1]),(left_pos[0]+length,left_pos[1]),
                                (right_pos[0],right_pos[1]),(left_pos[0],left_pos[1]+width)])
                self.obs_poly_set[obs_name] = poly
                self.rt_idx.insert(self.obs_index, poly.bounds)
                self.obs_index += 1
                self.rt_obs_set.append(poly)
                self.rt_obs_type.append('polygon')

            if obs_type == 'cen_rect':
                cen_pos = obs_charact['cen_pos']
                rect_size = obs_charact['size']
                poly = Polygon([(cen_pos[0]+rect_size[0],cen_pos[1]+rect_size[1]),(cen_pos[0]-rect_size[0],cen_pos[1]+rect_size[1]),
                                (cen_pos[0]-rect_size[0],cen_pos[1]-rect_size[1]),(cen_pos[0]+rect_size[0],cen_pos[1]-rect_size[1])])
                self.obs_poly_set[obs_name] = poly 
                self.rt_idx.insert(self.obs_index, poly.bounds)
                self.obs_index += 1
                self.rt_obs_set.append(poly)
                self.rt_obs_type.append('polygon')
        
            if obs_type == 'circ':
                self.obs_circ_set.append(obs_charact)
                cen_pos = obs_charact["cen_pos"]
                radius = obs_charact["radius"]
                min_x = cen_pos[0] - radius
                min_y = cen_pos[1] - radius
                max_x = cen_pos[0] + radius
                max_y = cen_pos[1] + radius
                self.rt_idx.insert(self.obs_index, (min_x, min_y, max_x, max_y))
                self.obs_index += 1
                self.rt_obs_set.append(obs_charact)
                self.rt_obs_type.append('circ')
                
            if obs_type == 'vert_points':
                points = obs_charact['points']
                poly = Polygon(points)
                self.obs_poly_set[obs_name] = poly

    def plt_workspace(self,ax,param={}):
        obs_color = param.get('obs_color', 'black')
        ax.set_xlim((self.map_range[0],self.map_range[1]))
        ax.set_ylim((self.map_range[2],self.map_range[3]))
        obs_circ_set = copy.deepcopy(self.obs_circ_set)

        for obs in self.dy_obs_circ_set:
            if obs.type == 'circ':
                obs_circ_set.append(obs.charact)

        for key in self.obs_poly_set.keys():
            color = obs_color
            x,y = self.obs_poly_set[key].exterior.xy
            polygon = Polygon_plt(np.column_stack((x, y)), closed = True, facecolor = color, edgecolor = color)
            ax.add_patch(polygon)
            ax.plot(x, y, color = 'black', linewidth = 1, zorder = 2)
        
        for circ_obs in obs_circ_set:
            obs_cen = circ_obs['cen_pos']
            obs_radius = circ_obs['radius']
            theta = np.linspace(0, 2*np.pi, 50)
            xc = obs_cen[0]
            yc = obs_cen[1]
            x = xc + obs_radius*np.cos(theta)
            y = yc + obs_radius*np.sin(theta)
            gray = [191/255, 191/255, 191/255]
            ax.fill(x, y, facecolor = gray, zorder = 1)
            ax.plot(x,y,color = 'black', linewidth = 1, zorder = 1)
        ax.set_aspect('equal', adjustable='box')
    
    def radomGeneratePosFromDict(self, pointN, shape_param = None, shape_flag = False):
        excluded_region = list(self.obs_poly_set.values())
        circ_excluded_region = copy.deepcopy(self.obs_circ_set)
        default_dist = 10e-2
        safe_radius = copy.deepcopy(default_dist)

        shape_type = ''
        if shape_param is not None:
            shape_type = shape_param['type']
            shape_charct = shape_param['charact']
            if shape_type == 'circ':
                safe_radius = safe_radius + shape_charct['radius']

        points = []
        max_iter = 100 * pointN
        iter_times = 0
        point_idx = 0
        while len(points) < pointN and iter_times <= max_iter:
            if shape_type == 'circ':
                x = round(np.random.uniform(self.map_range[0] + safe_radius, self.map_range[1] - safe_radius),2)
                y = round(np.random.uniform(self.map_range[2] + safe_radius, self.map_range[3] - safe_radius),2)
                point = Point(x, y)
                is_within_polygon = any(polygon.distance(point) <= (safe_radius) for polygon in excluded_region)
                is_within_circ = False
                for circ_obs in circ_excluded_region:
                    circ_obs_pos = np.array(circ_obs['cen_pos'])
                    circ_obs_radius = circ_obs['radius']
                    point_pos = np.array([x,y])
                    if np.linalg.norm(point_pos - circ_obs_pos) < (circ_obs_radius + safe_radius):
                        is_within_circ = True
                        break
                if not (is_within_polygon or is_within_circ):
                    if safe_radius > default_dist:
                        circ_excluded_region.append({'cen_pos':[x,y], 'radius':safe_radius})
                    points.append((x, y))
                    point_idx += 1
            elif shape_type == 'poly':
                poly_list = shape_charct['poly_list']
                safe_dist = shape_charct.get('safe_dist', 0.1)
                x = round(np.random.uniform(self.map_range[0] + safe_radius, self.map_range[1] - safe_radius),2)
                y = round(np.random.uniform(self.map_range[2] + safe_radius, self.map_range[3] - safe_radius),2)
                if shape_flag:
                    theta = round(np.random.uniform(-np.pi,np.pi),2)
                else:
                    theta = 0
                poly_shape = poly_list[point_idx % len(poly_list)]
                shape_poly = translate(rotate(poly_shape, angle = theta/np.pi*180, origin = (0, 0)), xoff = x, yoff = y)
                collision_flag = any(polygon.distance(shape_poly) <= (safe_dist) for polygon in excluded_region)
                if not collision_flag:
                    points.append((x, y, theta))
                    point_idx += 1
                    excluded_region.append(shape_poly)
            else:
                print(f"Unsupported shape type: {shape_type}")  
                break
            iter_times = iter_times + 1
        point_pos = np.array(points)
        return point_pos