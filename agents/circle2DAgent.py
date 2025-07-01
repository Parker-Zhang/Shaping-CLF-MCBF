"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         circle2DAgent.py
Description:    2D Circular Robot Class

Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
import numpy as np
import copy
from qpsolvers import solve_qp
from shapely.geometry import Point
import qpSWIFT
from utils.geometryLib import geometryLib
from method.CBF_DeadlockSolver import CBF_DeadlockSolver

class Circle2DAgent():
    def __init__(self):
        self.init_agent_param()
        self.init_cbf_param()
        self.geo_lib = geometryLib()
        
    """ ----------- initialization function ----------------- """
    def init_agent_param(self):
        self.cur_pos = np.array([0,0])
        self.tar_pos = np.array([0,0])
        self.vel = np.array([0,0])
        self.agent_r = 0.2
        self.sense_radius = np.inf
        self.comm_radius = np.inf
        self.vmax = 1.0
        self.nei_agent_pos = []
        self.nei_agent_vel = []

    def init_CLF_param(self):
        self.ellip_theta = 0
        self.ellip_s = [1,1]
        self.user_ellip_s = [1,1]
        self.user_ellip_theta = 0
        self.alpha = 2
        self.rho = 0.5

    def init_cbf_param(self):
        self.init_CLF_param()
        self.CLF_shaping_flag = False
        self.cbf_boundary_hold = {"agentCA":0.1, "obsCA":0.1}
        deadlock_solver_param = {"ellip_theta":self.ellip_theta, "ellip_s":self.ellip_s,}
        self.deadlock_solver = CBF_DeadlockSolver(deadlock_solver_param)
        self.deadlock_type = 1 # completely deadlock-free

    def init_workspace(self,worksapce):
        self.workspace = copy.deepcopy(worksapce)
    
    def update_state(self,dt):
        self.cur_pos = self.cur_pos + self.vel * dt

    """ ----------- plotting function ----------------- """    
    def plt_vec(self, ax, pos, vec, param):
        vec_color = param.get('vec_color', 'blue')
        zorder = param.get('zorder', 5)
        scale = param.get('scale', 1.0)
        width = param.get('width', 0.015)
        alpha = param.get('alpha', 0.5)

        x = pos[0]
        y = pos[1]
        v_mag = np.linalg.norm(vec)
        if v_mag < 10e-6:
            return
        vx = vec[0]/v_mag
        vy = vec[1]/v_mag
        ax.quiver(x, y, vx, vy, color = vec_color, angles='xy', scale_units='xy',
                scale=scale, width = width, linestyle='dashed',zorder=zorder, pivot = 'tip', alpha=alpha)
        
    def plt_key_vector(self, ax, param, pos=None):
        if pos is None:
            pos = self.cur_pos
        active_cbf_grad_set = param.get('active_cbf_grad_set', [])
        for cbf_grad in active_cbf_grad_set:
            self.plt_vec(ax, pos, np.array(cbf_grad), 
                         {'vec_color':'red', 'scale':1.5, 'width':0.01, 'zorder':10})
        stable_force = param.get('stable_force', np.zeros(2))
        self.plt_vec(ax, pos, -stable_force,
                     {'vec_color':'blue', 'scale':2, 'width':0.01, 'zorder':10})


    """    ----------- shaping-CLF-MCBF-QP controller-----------------"""
    def CLF_CBF_QP_controller(self):
        ctl_var_N = 3
        slack_index = 2
        
        P = np.diag(np.ones(ctl_var_N))
        q = np.zeros((ctl_var_N,1))
        g = np.empty((0,ctl_var_N))
        h = np.empty((0,1))

        pos_err = self.cur_pos - self.tar_pos
        self.pos_err = copy.deepcopy(pos_err)
        u_norm = np.zeros(2)
        
        # ------- Constructing CLF Constraints --------
        rotate_R = np.array([[np.cos(self.ellip_theta), -np.sin(self.ellip_theta)],
                             [np.sin(self.ellip_theta), np.cos(self.ellip_theta)],])
        ellip_H = np.diag([self.ellip_s[0]**2, self.ellip_s[1]**2])
        # clf_tmp = 0.5 * np.dot(pos_err, rotate_R @ ellip_H @ rotate_R.transpose() @ pos_err)
        clf = 0.5 * np.dot(pos_err, pos_err)  # Avoid CLF decay
        grad_tmp = rotate_R @ ellip_H @ rotate_R.transpose()
        cond = np.linalg.cond(grad_tmp)
        clf_grad = grad_tmp @ pos_err

        # Avoiding singularity
        if cond > 10e6:
            cross_product = np.cross(pos_err, clf_grad)
            now_direct = np.sign(cross_product)
            if now_direct != self.deadlock_solver.shaping_direct:
                clf_grad = -clf_grad
        
        clf_x_grad = copy.deepcopy(clf_grad)
        
        alpha = self.alpha
        rho = self.rho
        g_tmp = np.zeros((1,ctl_var_N))
        g_tmp[0,0:2] = copy.deepcopy(clf_grad)
        g_tmp[0,slack_index] = -1
        P[slack_index,slack_index] = 5
        h_tmp = -alpha * np.sign(clf) * abs(clf)**rho
        g = np.vstack([g,g_tmp])
        h = np.vstack([h,h_tmp])
        gamma_V = -h_tmp
        
        # ------- Constructing CBF constraints --------
        g_cbf, h_cbf, cbf_param = self.construct_CBF_constraints(ctl_var_N)
        g = np.vstack([g,g_cbf])
        h = np.vstack([h,h_cbf])

        # ------- Integrated deadlock detection and resolution module -----------
        deadlock_type = 1
        # If it is close to the target, the deadlock is no longer determined.
        if np.linalg.norm(pos_err) > 0.5:
            cbf_param.update({'g':g,'h':h,'P':P,'q':q,'u_norm':u_norm})
            cbf_param.update({'pos_err':pos_err,'gamma_V':gamma_V,'clf_x_grad':clf_x_grad,})
            deadlock_res = self.deadlock_solver.detect_deadlock(cbf_param)
            cbf_param.update(deadlock_res)
            deadlock_type = cbf_param['deadlock_type']
            # --------- Calling the CLF shaping module -----------
            if self.CLF_shaping_flag:
                ref_clf_grad = copy.deepcopy(pos_err)
                cbf_param.update({'ref_clf_grad':ref_clf_grad})
                ellip_theta_after, ellip_s_after, deadlock_type_ = self.deadlock_solver.CLF_shaping_module(cbf_param)
                self.ellip_theta = copy.deepcopy(ellip_theta_after)
                self.ellip_s = copy.deepcopy(ellip_s_after)
        else:
            self.ellip_theta = copy.deepcopy(self.user_ellip_theta)
            self.ellip_s = copy.deepcopy(self.user_ellip_s)
            self.deadlock_solver.ellip_theta = copy.deepcopy(self.ellip_theta)
            self.deadlock_solver.ellip_s = copy.deepcopy(self.ellip_s)
            cbf_param.update({'deadlock_type':deadlock_type, 'active_cbf_value':[],
                              'clf_cone_dist':np.inf, 'cone_theta': np.inf})

        self.deadlock_type = copy.deepcopy(deadlock_type)

        #------Calculate Control Inputs------
        constraint_N = g.shape[0]
        for i in range(constraint_N):
            A_ = g[i,0:-1]
            h[i,0] = h[i,0] - np.dot(u_norm, A_)
        v_max = copy.deepcopy(self.vmax)
        lb = [-v_max-u_norm[0], -v_max-u_norm[1], -np.inf]
        ub = [v_max-u_norm[0], v_max-u_norm[1], np.inf]
        lb = np.array(lb).flatten()
        ub = np.array(ub).flatten()
        # out = solve_qp(P = P, q = q, G = g, h = h, lb = lb, ub = ub, solver="cvxopt", qqp = True)
        out = self.QP_solver(P, q, g, h)
        
        if out is not None:
            self.vel = out[0:2] + u_norm
            self.slack = out[2]
        else:
            self.vel = np.zeros(2)
            self.slack = 0
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > v_max:
            self.vel = self.vel * v_max/vel_mag
        # print("vel: ", self.vel, "slack:", self.slack)
        cbf_param.update({'vel': self.vel, 'ellip_s': self.ellip_s,'ellip_theta': self.ellip_theta,})
        return cbf_param


    """ ------- Constructing CBF Constraints -------- """
    def construct_CBF_constraints(self, ctl_var_N):
        g = np.empty((0,ctl_var_N))
        h = np.empty((0,1))
        cbf_grad_set = []
        cbf_value = []
        cbf_boundary_hold_set = []
        beta_cbf = []
        cbf_type = []
        # CBF constraints for collision avoidance with circular obstacles
        for circ_obs in self.workspace.obs_circ_set:
            obs_cen = np.array(circ_obs['cen_pos'])
            obs_r = circ_obs['radius']
            obs_dist = np.linalg.norm(self.cur_pos-obs_cen) - (obs_r + self.agent_r)
            cbf = 0.5*np.linalg.norm(self.cur_pos-obs_cen)**2 - 0.5*(obs_r + self.agent_r)**2
            cbf_grad = self.cur_pos - obs_cen
            cbf_grad = np.round(cbf_grad, 4)

            if self.sense_radius >= obs_dist:
                g_tmp = np.zeros((1,ctl_var_N))
                g_tmp[0,0:2] = -cbf_grad
                beta = 1
                h_tmp = beta * cbf
                g = np.vstack([g,g_tmp])
                h = np.vstack([h,h_tmp])
                cbf_value.append(cbf)
                beta_cbf.append(-h_tmp)
                cbf_grad_set.append(cbf_grad)
                cbf_type.append("obsCA")
                cbf_boundary_hold_set.append(self.cbf_boundary_hold["obsCA"])

        # CBF constraints for collision avoidance with polygonal obstacles
        for poly_obs in self.workspace.obs_poly_set.values():
            point_shape = Point(self.cur_pos)
            dist, grad = self.geo_lib.get_point2poly_dist_grad(point_shape, poly_obs)
            obs_dist = dist - self.agent_r
            cbf = 0.5 * dist**2 - 0.5 * (self.agent_r)**2
            cbf_grad = np.round(grad[0:2], 4)
            if self.sense_radius >= obs_dist:
                g_tmp = np.zeros((1,ctl_var_N))
                g_tmp[0,0:2] = -cbf_grad
                beta = 1
                h_tmp = beta * cbf
                g = np.vstack([g,g_tmp])
                h = np.vstack([h,h_tmp])
                
                cbf_value.append(cbf)
                beta_cbf.append(-h_tmp)
                cbf_grad_set.append(cbf_grad)
                cbf_type.append("obsCA")
                cbf_boundary_hold_set.append(self.cbf_boundary_hold["obsCA"])

        # CBF constraints for collision avoidance with neighboring agents
        nei_agent_N = len(self.nei_agent_pos)
        for i in range(nei_agent_N):
            nei_agent_pos_ = self.nei_agent_pos[i]
            nei_agent_vel_ = self.nei_agent_vel[i]
            
            cbf = 0.5*np.linalg.norm(self.cur_pos-nei_agent_pos_)**2 - 0.5*(2*self.agent_r)**2
            cbf_grad = self.cur_pos-nei_agent_pos_
            cbf_grad = np.round(cbf_grad, 4)
            
            g_tmp = np.zeros((1,ctl_var_N))
            g_tmp[0,0:2] = -cbf_grad
            
            beta = 1
            h_tmp = beta * cbf # - np.dot(cbf_grad, nei_agent_vel_)
            
            g = np.vstack([g,g_tmp])
            h = np.vstack([h,h_tmp])
            cbf_value.append(cbf)
            beta_cbf.append(-h_tmp)
            cbf_grad_set.append(cbf_grad)
            cbf_type.append("agentCA")
            cbf_boundary_hold_set.append(self.cbf_boundary_hold["agentCA"])

        cbf_param = {'cbf_grad_set':cbf_grad_set,'cbf_value':cbf_value,'cbf_type':cbf_type,
                     'cbf_boundary_hold_set':cbf_boundary_hold_set, 'beta_cbf':beta_cbf, }
        return g,h,cbf_param

    def QP_solver(self, P, q, g, h, lb = None, ub = None):
        if lb is not None or ub is not None:
            out = solve_qp(P = P, q = q, G = g, h = h, lb = lb, ub = ub, solver="cvxopt")
            return out
        else:
            opts = {'MAXITER': 30, 'VERBOSE': 0, 'OUTPUT': 0}
            res = qpSWIFT.run(q.flatten(),h.flatten(),P,g,opts=opts)
            out = res['sol']
            return out