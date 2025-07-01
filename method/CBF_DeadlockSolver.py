"""
================================================================================
Project:        Shaping-CLF-MCBF
Module:         CBF_DeadlockSolver.py
Description:    General tools for deadlock detection and reactive avoidance for CLF-MCBF-QP controllers
                Main function
                - detect_deadlock
                - CLF_shaping_module

Author:         Zhenwei Zhang
Contact:        zhenweizhang@hust.edu.cn
GitHub:         https://github.com/Parker-Zhang/Shaping-CLF-MCBF.git

Created on:     2025-06-30
License:        MIT License
================================================================================
"""
import numpy as np
import qpSWIFT
from qpsolvers import solve_qp
import copy
from scipy.optimize import minimize
import math

class CBF_DeadlockSolver:
    def __init__(self, param):
        self.init_CLF_shaping_module(param)

    def init_CLF_shaping_module(self, param):
        self.ellip_s = param['ellip_s']
        self.ellip_theta = param['ellip_theta']
        self.user_ellip_s = param['ellip_s']
        self.user_ellip_theta = param['ellip_theta']

        self.shaping_direct = 1
        self.direction_hold = 0
        self.direction_hold_max = 20 # 40
        self.shape_hold_step = 0
        self.shape_hold_max = 10
        self.reshape_gain_hold = 0
        self.reshape_gain_hold_max = 20
        self.dist_hold = 1.5
        self.angle_hold = 20

    def detect_deadlock(self, param):
        P = np.array(param['P'])
        pos_err = np.array(param['pos_err'])
        u_norm = np.array(param['u_norm'])
        g_tmp = np.array(param['g'])
        h_tmp = np.array(param['h'])
        q = np.array(param['q'])
        gamma_V = param['gamma_V']
        clf_x_grad = np.array(param['clf_x_grad'])
        cbf_grad_set = param['cbf_grad_set']
        cbf_value = param['cbf_value']
        cbf_boundary_hold_set = param['cbf_boundary_hold_set']
        cbf_type = param['cbf_type']
        
        deadlock_type = 1
        # 1) get active CBF constraints
        active_cbf_hold = 10e-3
        constraint_N = g_tmp.shape[0]
        
        for i in range(constraint_N):
            A_ = g_tmp[i,0:-1]
            h_tmp[i,0] = h_tmp[i,0] - np.dot(u_norm, A_)
        out_tmp = self.QP_solver(P, q, g_tmp, h_tmp)
        if out_tmp is None:
            active_flag = [True for i in range(constraint_N-1)]
            active_value = [0 for i in range(constraint_N-1)]
        else:
            active_value = np.dot(g_tmp[1:,:], out_tmp) - h_tmp[1:].flatten()
            active_flag = [abs(i) < active_cbf_hold for i in active_value]

        # 2) get active CBF gradients
        active_cbf_grad = np.empty((0,2))
        active_cbf_grad_set = []
        active_cbf_grad_type = []
        active_cbf = []
        index = 0
        for flag in active_flag:
            grad = cbf_grad_set[index]
            if flag and np.linalg.norm(grad)!= 0:
                if cbf_value[index] < cbf_boundary_hold_set[index]:
                    active_cbf_grad = np.vstack([active_cbf_grad, grad])
                    active_cbf.append(cbf_value[index])
                    active_cbf_grad_set.append(grad)
                    active_cbf_grad_type.append(cbf_type[index])
            index = index + 1

        # 3) calculate the stablilizing force
        G = np.diag([1,1])
        sys_stable_force = np.linalg.pinv(P[-1,-1] * gamma_V * G) @ u_norm
        stable_force = -(sys_stable_force - clf_x_grad)
        ref_clf_grad = copy.deepcopy(pos_err)
        ref_stable_force = - (sys_stable_force - ref_clf_grad)
        
        deadlock_flag = False
        clf_cone_dist = np.inf
        cone_theta = np.inf
        active_cbf_N = active_cbf_grad.shape[0]
        
        if active_cbf_N > 0 and np.linalg.norm(pos_err) > 10e-3:
            dist_cone = self.fast_cal_grad_cone_dist(stable_force, active_cbf_grad)
            clf_cone_dist = np.linalg.norm(dist_cone)
            flag = self.decide_conical_comb(active_cbf_grad, stable_force, )
            if flag:
                deadlock_flag = True
                if np.linalg.norm(stable_force) < 10e-3:
                    self.reset_clf_shape()
            else:
                deadlock_type = 2
            
        # 4) get boundary of conical hull
        boundary = []
        if active_cbf_N > 2:
            test_point = list(active_cbf_grad)
            test_point.append(np.array([0,0]))
            convex_polygon = self.convex_hull([list(vec) for vec in list(test_point)])
            deadlock_type_flag = any(np.array_equal(row, np.array([0,0])) for row in convex_polygon)
            if deadlock_type_flag:
                for k in range(active_cbf_N):
                    test_cbf = active_cbf_grad[k,:]
                    test_cbf_mat = np.delete(active_cbf_grad, k, axis = 0)
                    boundary_flag = self.decide_conical_comb(test_cbf_mat, test_cbf)
                    if not boundary_flag:
                        boundary.append(test_cbf)
        else:
            boundary = [active_cbf_grad[k,:] for k in range(active_cbf_N)]

        # 5) determine deadlock type
        if deadlock_flag:
            if len(boundary) > 0:
                deadlock_type = 3
            else:
                deadlock_type = 4
                clf_cone_dist = 0
                cone_theta = 0
        else:
            if len(boundary) > 0:
                deadlock_type = 2
            else:
                deadlock_type = 1

        deadlock_detect_res = {'deadlock_type':deadlock_type, 'boundary':boundary, 
                               'stable_force':stable_force, 'sys_stable_force':sys_stable_force,
                               'active_cbf_grad':active_cbf_grad,'clf_cone_dist':clf_cone_dist, 'cone_theta':cone_theta,
                               'active_cbf':active_cbf,'ref_stable_force':ref_stable_force, 
                               'active_cbf_grad_set': active_cbf_grad_set, 'active_cbf_value': active_value}
        return deadlock_detect_res
    

    """   --------------- CLF shaping module ----------------"""
    def CLF_shaping_module(self,CLF_shaping_param):
        deadlock_type = CLF_shaping_param["deadlock_type"]
        boundary = CLF_shaping_param["boundary"]
        pos_err = CLF_shaping_param["pos_err"]
        stable_force = CLF_shaping_param["stable_force"]
        sys_stable_force = CLF_shaping_param["sys_stable_force"]
        ref_clf_grad = CLF_shaping_param["ref_clf_grad"]
        ref_stable_force = - (sys_stable_force - ref_clf_grad)
        active_cbf_grad = CLF_shaping_param["active_cbf_grad"]
        cbf_grad_set = CLF_shaping_param["cbf_grad_set"]
        
        if len(boundary) > 2:
            boundary = self.extract_opposites(boundary)

        if np.linalg.norm(pos_err)!= 0:
            pos_err_unit = pos_err / np.linalg.norm(pos_err)
        else:
            pos_err_unit = np.zeros(2)
        
        ellip_s1_after = copy.deepcopy(self.ellip_s[0])
        ellip_s2_after = copy.deepcopy(self.ellip_s[1])
        ellip_theta_after = copy.deepcopy(self.ellip_theta)
        
        # Handling of deadlocks by type
        dist_hold = copy.deepcopy(self.dist_hold)
        angle_hold = copy.deepcopy(self.angle_hold)
        if deadlock_type == 3:
            # handle weak deadlock
            stable_force_dist_list = []
            stable_force_dist_vec_list = []
            stable_force_proj_vec_list = []
            stable_force_shaping_list = []
            
            for boun_vec in boundary:
                if np.dot(boun_vec, pos_err_unit) < 0:
                    continue
                
                stable_force_proj_unit = boun_vec/np.linalg.norm(boun_vec)
                stable_force_proj = abs(np.dot(stable_force_proj_unit, stable_force)) * stable_force_proj_unit
                stable_force_dist_vec = stable_force_proj - stable_force
                if np.linalg.norm(stable_force_dist_vec) == 0:
                    stable_force_dist_vec = np.array([stable_force_proj[1], -stable_force_proj[0]])/np.linalg.norm(stable_force_proj)
        
                dist_hold_angle = np.tan(angle_hold*np.pi/180) *  np.linalg.norm(stable_force_proj)
                dist_hold = min(dist_hold, dist_hold_angle)
                stable_force_change_norm = copy.deepcopy(dist_hold)
                stable_force_change_unit = stable_force_dist_vec / np.linalg.norm(stable_force_dist_vec)
                stable_force_shaping_candi = stable_force_proj + stable_force_change_unit * stable_force_change_norm
                stable_force_shaping_candi = np.linalg.norm(pos_err)*stable_force_shaping_candi/np.linalg.norm(stable_force_shaping_candi)
                
                half_plane_b = - np.dot(sys_stable_force, pos_err_unit)
                half_plane_a = np.dot(stable_force_shaping_candi, pos_err_unit)
                factor = max(1.2* half_plane_b/half_plane_a, 1)
                stable_force_shaping_candi = factor * stable_force_shaping_candi
                
                
                stable_force_dist = np.linalg.norm(ref_stable_force - stable_force_shaping_candi) - dist_hold
                
                stable_force_dist_list.append(stable_force_dist)
                stable_force_dist_vec_list.append(stable_force_dist_vec)
                stable_force_proj_vec_list.append(stable_force_proj)
                stable_force_shaping_list.append(stable_force_shaping_candi)
                
                if len(boundary) == 1:
                    stable_force_shaping_candi2 = stable_force_proj - stable_force_change_unit * stable_force_change_norm
                    stable_force_shaping_candi2 = np.linalg.norm(pos_err)*stable_force_shaping_candi2/np.linalg.norm(stable_force_shaping_candi2)
                    half_plane_b = - np.dot(sys_stable_force, pos_err_unit)
                    half_plane_a = np.dot(stable_force_shaping_candi, pos_err_unit)
                    factor = max(1.2* half_plane_b/half_plane_a, 1)
                    stable_force_shaping_candi2 = factor * stable_force_shaping_candi2
                    stable_force_dist = np.linalg.norm(ref_stable_force - stable_force_shaping_candi2) - dist_hold
                    
                    stable_force_dist_list.append(stable_force_dist)
                    stable_force_dist_vec_list.append(stable_force_dist_vec)
                    stable_force_proj_vec_list.append(stable_force_proj)
                    stable_force_shaping_list.append(stable_force_shaping_candi2)
                    
            if len(stable_force_shaping_list) > 0:
                clf_cone_dist = -min(stable_force_dist_list)
                change_flag = False
                if self.shaping_direct == 0:
                    bound_index = stable_force_dist_list.index(min(stable_force_dist_list))
                    stable_force_shaping = stable_force_shaping_list[bound_index]
                    cross_product = np.cross(stable_force, stable_force_shaping)
                    self.shaping_direct = np.sign(cross_product)
                    change_flag = True
                else:
                    clock_sign = []
                    for stable_force_shaping_ in stable_force_shaping_list:
                        clock_tmp = np.cross(stable_force, stable_force_shaping_)
                        clock_sign.append(np.dot(self.shaping_direct, clock_tmp))
                    bound_index_set = [i for i in range(len(clock_sign)) if clock_sign[i] > 0]
                    if len(bound_index_set) > 0:
                        bound_index = bound_index_set[0]
                        change_flag = True
                    else:
                        self.revise_change_direct()
                    if change_flag:
                        stable_force_shaping = stable_force_shaping_list[bound_index]

                if change_flag:
                    clf_grad_shaping = stable_force_shaping + sys_stable_force
                    origin_grad = copy.deepcopy(ref_clf_grad)
                    target_grad = copy.deepcopy(clf_grad_shaping)
                    opt_theta1, opt_s1, opt_s2, opt_err = self.optimize_shape_param(origin_grad, target_grad)

                    rotate_R = np.array([[np.cos(opt_theta1), -np.sin(opt_theta1)], [np.sin(opt_theta1), np.cos(opt_theta1)]])
                    ellip_H = np.array([[opt_s1**2, 0], 
                                        [0, opt_s2**2],])
                    new_clf_grad = rotate_R @ ellip_H @ rotate_R.T @ pos_err
                    new_clf_x_grad = new_clf_grad[0:2]
                    
                    new_stable_force = -( sys_stable_force - new_clf_x_grad )
        
                    new_dist_cone = self.fast_cal_grad_cone_dist(new_stable_force, active_cbf_grad)
                    if np.linalg.norm(new_dist_cone) < 10e-3:
                        # print('still weak deadlock!')
                        self.revise_change_direct()
                        clf_cone_dist = -np.pi
                        deadlock_type = 4
                    else:
                        ellip_s1_after = copy.deepcopy(opt_s1)
                        ellip_s2_after = copy.deepcopy(opt_s2)
                        ellip_theta_after = copy.deepcopy(opt_theta1)
            else:
                clf_cone_dist = -np.pi
                deadlock_type = 4
                
        elif deadlock_type == 4:
            clf_cone_dist = -np.inf
        else:
            k1 = -5.0
            k2 = -5.0
            
            if deadlock_type == 1:
                # if completely deadlock-free, rebound to the user-defined shape
                dot_s1 = k1*(self.ellip_s[0] - self.user_ellip_s[0])
                dot_s2 = k1*(self.ellip_s[1] - self.user_ellip_s[1])
                dot_theta = k2*(self.ellip_theta - self.user_ellip_theta)
                
                cbf_grad_np = np.empty((0,2))
                for grad in cbf_grad_set:
                    cbf_grad_np = np.vstack([cbf_grad_np,grad])
                if len(cbf_grad_set) > 0:
                    dist = self.fast_cal_grad_cone_dist(stable_force, cbf_grad_np)
                    clf_cone_dist = np.linalg.norm(dist)
                
            elif deadlock_type == 2:
                # if resitrcted deadlock-free, construct virtual CBF 
                angle_force_cone = [np.dot(stable_force, boun_vec) for boun_vec in boundary]
                if max(angle_force_cone) >= 0:
                    k1 = 0
                    k2 = 0
                    if len(boundary) == 1:
                        k1,k2 = self.decide_reshape_gain()
                
                dot_s1_ref = k1*(self.ellip_s[0] - self.user_ellip_s[0])
                dot_s2_ref = k1*(self.ellip_s[1] - self.user_ellip_s[1])
                dot_theta_ref = k2*(self.ellip_theta - self.user_ellip_theta)
            
                rotate_R = np.array([[np.cos(self.ellip_theta), -np.sin(self.ellip_theta)],
                                      [np.sin(self.ellip_theta), np.cos(self.ellip_theta)],])
                ellip_H = np.diag([self.ellip_s[0]**2, self.ellip_s[1]**2])
                sigma = np.array([[0,-1],[1,0]])
                
                H_grad_s1 = np.array([[2*self.ellip_s[0],0],[0,0]])
                H_grad_s2 = np.array([[0,0],[0,2*self.ellip_s[1]]])
                
                nabla_clf_grad_s1 = 2 * rotate_R @ H_grad_s1 @ rotate_R.transpose() @ pos_err
                nabla_clf_grad_s2 = 2 * rotate_R @ H_grad_s2 @ rotate_R.transpose() @ pos_err
                nabla_clf_grad_theta = 2*(rotate_R @ sigma @ ellip_H @ rotate_R.transpose() +
                                          rotate_R @ ellip_H @ sigma.transpose() @ rotate_R.transpose()) @ pos_err
                nabla_clf_grad_s1 = np.round(nabla_clf_grad_s1, 4)
                nabla_clf_grad_s2 = np.round(nabla_clf_grad_s2, 4)
                nabla_clf_grad_theta = np.round(nabla_clf_grad_theta, 4)

                Hess_x = 2 * rotate_R @ ellip_H @ rotate_R.transpose()
                
                # constrcu virtual CBF
                optimal_d = self.fast_cal_grad_cone_dist(stable_force, active_cbf_grad)
                shaping_param_N = 3
                P_shaping = np.diag([0.1,1,1])
                
                q_shaping = np.zeros((shaping_param_N,1))
                g_shaping = np.empty((0,shaping_param_N))
                h_shaping = np.empty((0,1))
                
                g_shaping_tmp = np.zeros((1,shaping_param_N))
                dist_grad_clf = -copy.deepcopy(optimal_d)
                dist_grad_clf_unit = dist_grad_clf/np.linalg.norm(dist_grad_clf)
                
                g_shaping_tmp[0,0] = -np.dot(dist_grad_clf_unit, nabla_clf_grad_theta)
                g_shaping_tmp[0,1] = -np.dot(dist_grad_clf_unit, nabla_clf_grad_s1)
                g_shaping_tmp[0,2] = -np.dot(dist_grad_clf_unit, nabla_clf_grad_s2)
                
                dist_hold_angle = np.linalg.norm(stable_force)*np.sin(angle_hold*np.pi/180)
                dist_hold = min(dist_hold, dist_hold_angle)
                
                h_shaping_tmp = 3*(np.linalg.norm(dist_grad_clf)**2 - dist_hold**2) \
                    - g_shaping_tmp[0,0]*dot_theta_ref - g_shaping_tmp[0,1]*dot_s1_ref \
                        - g_shaping_tmp[0,2]*dot_s2_ref # + np.dot(dist_grad_clf_unit, Hess_x @ self.vel)
                
                g_shaping = np.vstack([g_shaping,g_shaping_tmp])
                h_shaping = np.vstack([h_shaping,h_shaping_tmp])
                    
                try:
                    out_shaping = self.QP_solver(P_shaping, q_shaping, g_shaping, h_shaping)
                    if out_shaping is None:
                        out_shaping = np.zeros(shaping_param_N)
                        out_shaping[0] = -dot_theta_ref
                        out_shaping[1] = -dot_s1_ref
                        out_shaping[2] = -dot_s2_ref
                except:
                    out_shaping = np.zeros(shaping_param_N)
                    out_shaping[0] = -dot_theta_ref
                    out_shaping[1] = -dot_s1_ref
                    out_shaping[2] = -dot_s2_ref
                    self.shaping_direct = 0
                    
                dot_theta = out_shaping[0] + dot_theta_ref
                dot_s1 = out_shaping[1] + dot_s1_ref
                dot_s2 = out_shaping[2] + dot_s2_ref

            dt = 0.05
            ellip_theta_after = self.ellip_theta + dt * dot_theta
            ellip_s1_after = self.ellip_s[0] + dt * dot_s1
            ellip_s2_after = self.ellip_s[1] + dt * dot_s2
        
        ellip_s_after = [ellip_s1_after, ellip_s2_after]
        self.ellip_s = copy.deepcopy(ellip_s_after)
        self.ellip_theta = copy.deepcopy(ellip_theta_after)
        
        return ellip_theta_after, ellip_s_after, deadlock_type

    ''' --------------------- function library (math.) --------------------- '''
    def QP_solver(self, P, q, g, h, lb = None, ub = None):
        if lb is not None or ub is not None:
            out = solve_qp(P = P, q = q, G = g, h = h, lb = lb, ub = ub, solver="cvxopt")
            return out
        else:
            opts = {'MAXITER': 30, 'VERBOSE': 0, 'OUTPUT': 0}
            res = qpSWIFT.run(q.flatten(),h.flatten(),P,g,opts=opts)
            out = res['sol']
            return out

    def fast_cal_grad_cone_dist(self, clf_vec, cbf_mat):
        def objective_function(a, clf_vec, cbf_mat):
            error = np.sum((clf_vec - cbf_mat.T @ a)**2)
            return error
        initial_guess = np.ones(cbf_mat.shape[0])
        constraints = ({'type': 'ineq', 'fun': lambda a: a})
        result = minimize(objective_function, initial_guess, args=(clf_vec, cbf_mat),
                        constraints=constraints)
        optimized_a = result.x
        opt_value = result.fun
        final_d = cbf_mat.T @ optimized_a - clf_vec
        return final_d
    
    def decide_conical_comb(self, base_vecs, test_vec):
        flag = False
        qp_var_N = base_vecs.shape[0]
        
        if qp_var_N > 1:
            eff = np.zeros(qp_var_N)
            A = base_vecs.T
            b = test_vec
            G = -np.eye(qp_var_N)
            h = np.zeros(qp_var_N)
            opts = {'MAXITER': 30, 'VERBOSE': 0, 'OUTPUT': 0}
            res = qpSWIFT.run(np.zeros(qp_var_N), h, np.eye(qp_var_N), G, A, b, opts)
            if res['sol'] is not None:
                eff_value = res['sol']
                if np.all(eff_value >= 0) and np.linalg.norm(A.dot(eff_value) - b, ord=2) < 1e-6:
                    flag = True
        else:
            base_angle = math.atan2(base_vecs[0][0], base_vecs[0][1])
            test_angle = math.atan2(test_vec[0], test_vec[1])
            if abs(base_angle - test_angle) < 0.01:
                flag = True
        return flag

    def convex_hull(self, points):
        pts = sorted(set(map(tuple, points)))
        if len(pts) <= 1:
            return np.array(pts)
        
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return np.array(hull)


    def extract_opposites(self, boundary, tol=1e-6):
        n = len(boundary)
        units = []
        for v in boundary:
            norm = np.linalg.norm(v)
            if norm < tol:
                units.append(None)
            else:
                units.append(v / norm)

        opposite_idx = set()
        for i in range(n):
            ui = units[i]
            if ui is None: 
                continue
            for j in range(i+1, n):
                uj = units[j]
                if uj is None:
                    continue
                if abs(np.dot(ui, uj) + 1) < tol:
                    opposite_idx.add(i)
                    opposite_idx.add(j)
        return [boundary[i] for i in sorted(opposite_idx)]

    def rotate_scale_objective(self, parameters, origin_grad, target_grad):
        theta, s1, s2 = parameters
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H = np.array([[s1 ** 2, 0], [0, s2 ** 2]])
        vec = np.dot(R, np.dot(H, np.dot(R.T, origin_grad)))
        error = np.linalg.norm(vec - target_grad)
        return error
    
    def optimize_shape_param(self, origin_grad, target_grad):
        initial_params = [0, 1, 1]
        theta_bounds = (-np.pi/2, np.pi/2)
        s_bounds = (0, 10)
        bounds = (theta_bounds, s_bounds, s_bounds)
        result = minimize(self.rotate_scale_objective, initial_params,
                        args=(origin_grad, target_grad), bounds=bounds, method='L-BFGS-B')   
        opt_theta1, opt_s1, opt_s2 = result.x
        opt_value = result.fun
        return opt_theta1, opt_s1, opt_s2, opt_value

    def reset_clf_shape(self):
        self.shape_hold_step = self.shape_hold_step + 1
        if self.shape_hold_step >= self.shape_hold_max:
            self.ellip_theta = copy.deepcopy(self.user_ellip_theta)
            self.ellip_s = copy.deepcopy(self.user_ellip_s)
            self.shape_hold_step = 0

    def revise_change_direct(self):
        self.direction_hold = self.direction_hold + 1
        if self.direction_hold >= self.direction_hold_max:
            self.shaping_direct = -self.shaping_direct
            self.direction_hold = 0

    def decide_reshape_gain(self):
        self.reshape_gain_hold += 1
        if self.reshape_gain_hold < self.reshape_gain_hold_max:
            k1 = 0
            k2 = 0
        elif self.reshape_gain_hold > 2*self.reshape_gain_hold_max:
            self.reshape_gain_hold = 0
            k1 = 0
            k2 = 0
        else:
            k1 = -5.0
            k2 = -5.0
        return k1, k2