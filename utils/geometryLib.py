"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         geometryLib.py
Description:    Functions to compute distances and distance gradients between geometries.

Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as Polygon_plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.ops import nearest_points

class geometryLib():
    def get_poly_distance_grad(self, poly1_pose, poly1_shape, poly2_shape, delta = 0.01, dtheta = 0.1):
        dangle = dtheta/np.pi * 180
        distance = poly2_shape.distance(poly1_shape)
        gradient_x = (poly2_shape.distance(translate(poly1_shape, xoff = delta, yoff = 0)) - poly2_shape.distance(poly1_shape))/delta
        gradient_y = (poly2_shape.distance(translate(poly1_shape, xoff = 0, yoff = delta)) - poly2_shape.distance(poly1_shape))/delta
        gradient_theta = (poly2_shape.distance(rotate(poly1_shape, angle = dangle, origin = (poly1_pose[0],poly1_pose[1]))) - poly2_shape.distance(poly1_shape))/(dtheta)
        return distance, [gradient_x,gradient_y,gradient_theta]
    
    def get_poly_distance_grad2(self, poly1_pose, poly1_shape, poly2_shape, delta = 0.01, dtheta = 0.1):
        dangle = dtheta/np.pi * 180
        point1, point2 = nearest_points(poly1_shape, poly2_shape)
        x1, y1 = point1.xy
        x2, y2 = point2.xy
        
        distance = np.sqrt((x1[0]-x2[0])**2 + (y1[0]-y2[0])**2)
        if distance > 0:
            gradient_x = (x1[0] - x2[0])/distance
            gradient_y = (y1[0] - y2[0])/distance
        else:
            gradient_x = (poly2_shape.distance(translate(poly1_shape, xoff = delta, yoff = 0)) - poly2_shape.distance(poly1_shape))/delta
            gradient_y = (poly2_shape.distance(translate(poly1_shape, xoff = 0, yoff = delta)) - poly2_shape.distance(poly1_shape))/delta
            
        gradient_theta = (poly2_shape.distance(rotate(poly1_shape, angle = dangle, origin = (poly1_pose[0],poly1_pose[1]))) - poly2_shape.distance(poly1_shape))/(dtheta)
        return distance, [gradient_x,gradient_y,gradient_theta]
    
    def get_point_poly_distance_grad(self, poly_pose, poly_shape, point, delta = 0.01, dtheta = 0.3):
        dangle = dtheta/np.pi * 180
        distance = poly_shape.distance(point)
        gradient_x = (point.distance(translate(poly_shape, xoff = delta, yoff = 0)) - point.distance(poly_shape))/delta
        gradient_y = (point.distance(translate(poly_shape, xoff = 0, yoff = delta)) - point.distance(poly_shape))/delta
        gradient_theta = (point.distance(rotate(poly_shape, angle = dangle, origin = (poly_pose[0],poly_pose[1]))) - point.distance(poly_shape))/(dtheta)
        return distance, [gradient_x,gradient_y,gradient_theta]
    
    def get_point2poly_dist_grad(self, point, poly_shape):
        # distance = poly_shape.distance(point)
        point1, point2 = nearest_points(point, poly_shape)
        x1, y1 = point1.xy
        x2, y2 = point2.xy
        distance = np.sqrt((x1[0]-x2[0])**2 + (y1[0]-y2[0])**2)
        if distance > 0:
            gradient_x = (x1[0] - x2[0])/distance
            gradient_y = (y1[0] - y2[0])/distance
            round(gradient_x, 4)
            round(gradient_y, 4)
        else:
            gradient_x = 0
            gradient_y = 0
        return distance, [gradient_x,gradient_y]


    def shape2D_transformation(self, rigidbody, pose):
        rigidbody = np.asarray(rigidbody)
        assert rigidbody.shape[1] == 2, "rigidbody must be of shape (N, 2)"
        
        theta = pose[2]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
        t = np.array([pose[0], pose[1]])
        transformed = rigidbody @ R.T + t  # (N,2) @ (2,2).T + (2,) = (N,2)
        
        return transformed
            
            
        
        
        