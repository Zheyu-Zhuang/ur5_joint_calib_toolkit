#!/usr/bin/env python
import sys
import math
from math import sin, cos, pi, acos, asin, atan2
import numpy as np


GRASP_OFFSET = 0.08

class UR5Kin:
    """
    This Class states the forward kinematics of UR5 wrt World Frame.

    it provides several instance methods:

    - self.get_DH_mat(theta, d, alpha, a, is_pd=False) --> (a 4x4 np.array)
        if is_pd = True, it returns the pd of corresponding DH_matrix
        returns DH_matrix otherwise

    - self.get_pd_X_H(joint_config (i.e. v), joint_idx) --> (a 4x4 np.array):
        pd of hand pose wrt a specific joint

    - self.joint_poses(joint_config) --> (a 6x4x4 np.array):
        each [i, :, :] of the output is a joint pose of joint 'i'

    - self.get_veced_joint_poses(joint_config) --> (a 1x72 np.array):
        vertorised joint poses with out [0, 0, 0, 1].
    """

    # define UR5_Params

    def __init__(self):
        self.a = np.array([0, 0.42500, 0.39225, 0, 0, 0])
        self.d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
        self.alpha = np.array([-pi / 2, 0, 0, -pi / 2, pi / 2, 0])
        # self.H_base = np.array([[0.707, 0.707, 0, 0],
        #                         [-0.707, 0.707, 0, 0],
        #                         [0, 0, 1, 0.938],
        #                         [0, 0, 0, 1]])
        self.H_base = np.eye(4)

    def get_DH_mat(self, theta, d, alpha, a, is_pd=False):
        if is_pd:
            pd_H = np.array([[-sin(theta), -cos(theta) * cos(alpha),
                              cos(theta) * sin(alpha), -a * sin(theta)],
                             [cos(theta), -sin(theta) * cos(alpha),
                              sin(theta) * sin(alpha), a * cos(theta)],
                             [0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0]])
            return pd_H
        else:
            H = np.array([[cos(theta), -sin(theta) * cos(alpha),
                           sin(theta) * sin(alpha), a * cos(theta)],
                          [sin(theta), cos(theta) * cos(alpha),
                           -cos(theta) * sin(alpha), a * sin(theta)],
                          [0.0, sin(alpha), cos(alpha), d],
                          [0.0, 0.0, 0.0, 1.0]])
            return H

    def get_pd_X_H(self, v, joint_idx):
        # for the ease of coding, the name of pd dh matrix remains the same as
        # is dh form.
        I = np.eye(6)
        pd_flag_vec = I[joint_idx, :]
        H01 = self.H_base.dot(
            self.get_DH_mat(v[0, 0], self.d[0], self.alpha[0], self.a[0],
                            pd_flag_vec[0]))
        H12 = self.get_DH_mat(v[1, 0], self.d[1], self.alpha[1], self.a[1],
                              pd_flag_vec[1])
        H23 = self.get_DH_mat(v[2, 0], self.d[2], self.alpha[2], self.a[2],
                              pd_flag_vec[2])
        H34 = self.get_DH_mat(v[3, 0], self.d[3], self.alpha[3], self.a[3],
                              pd_flag_vec[3])
        H45 = self.get_DH_mat(v[4, 0], self.d[4], self.alpha[4], self.a[4],
                              pd_flag_vec[4])
        H56 = self.get_DH_mat(v[5, 0], self.d[5], self.alpha[5], self.a[5],
                              pd_flag_vec[5])
        H02 = H01.dot(H12)
        H03 = H02.dot(H23)
        H04 = H03.dot(H34)
        H05 = H04.dot(H45)
        H06 = H05.dot(H56)
        pd_X_H = H06
        return pd_X_H

    def dewedge(self, omega_x):
        return np.array((omega_x[2, 1], omega_x[0, 2], omega_x[1, 0]))

    def cmpt_jacobian_dumb(self, v):
        X = self.get_ee_pose(v)
        jacobian = np.zeros((6, 6))
        for joint_idx in range(6):
            pd_X = self.get_pd_X_H(v, joint_idx)
            jacobian[0:3, joint_idx] = pd_X[0:3, 3]
            omega_x = pd_X.dot(np.linalg.inv(X))[0:3, 0:3]
            jacobian[3:6, joint_idx] = self.dewedge(omega_x)
        return jacobian

    def get_joint_poses(self, v):
        H01 = self.H_base.dot(self.get_DH_mat(v[0, 0], self.d[0], self.alpha[0],
                                              self.a[0]))
        H12 = self.get_DH_mat(v[1, 0], self.d[1], self.alpha[1], self.a[1])
        H23 = self.get_DH_mat(v[2, 0], self.d[2], self.alpha[2], self.a[2])
        H34 = self.get_DH_mat(v[3, 0], self.d[3], self.alpha[3], self.a[3])
        H45 = self.get_DH_mat(v[4, 0], self.d[4], self.alpha[4], self.a[4])
        H56 = self.get_DH_mat(v[5, 0], self.d[5], self.alpha[5], self.a[5])
        H02 = H01.dot(H12)
        H03 = H02.dot(H23)
        H04 = H03.dot(H34)
        H05 = H04.dot(H45)
        H06 = H05.dot(H56)
        joint_poses = np.array([H01, H02, H03, H04, H05, H06])
        return joint_poses

    def cmpt_jacobian(self, v, ref='world'):
        joint_poses = self.get_joint_poses(v)
        J = np.zeros((6, 6))
        p = joint_poses[5][0:3, 3]
        for i in range(0, 6):
            if i == 0:
                z_i_1 = np.array([0, 0, 1])
                p_i_1 = np.zeros(3)
            else:
                z_i_1 = joint_poses[i - 1][0:3, 2]
                p_i_1 = joint_poses[i - 1][0:3, 3]
            J[0:3, i] = np.cross(z_i_1, (p - p_i_1))
            J[3:6, i] = z_i_1
        if ref == 'ee':
            X_H = joint_poses[5]
            R_ee = X_H[0:3, 0:3]
            R_compound = np.zeros((6, 6))
            R_compound[0:3, 0:3], R_compound[3:6, 3:6] = R_ee, R_ee
            J = R_compound.transpose().dot(J)
        elif ref == 'world':
            pass
        else:
            print('Invalid Reference frame, returning J in inertia frame')
        return J

    def cmpt_jacobian_wrt_ee(self, v):
        J = self.cmpt_jacobian(v)

    def get_ee_pose(self, v):
        H01 = self.H_base.dot(self.get_DH_mat(v[0, 0], self.d[0], self.alpha[0],
                                              self.a[0]))
        H12 = self.get_DH_mat(v[1, 0], self.d[1], self.alpha[1], self.a[1])
        H23 = self.get_DH_mat(v[2, 0], self.d[2], self.alpha[2], self.a[2])
        H34 = self.get_DH_mat(v[3, 0], self.d[3], self.alpha[3], self.a[3])
        H45 = self.get_DH_mat(v[4, 0], self.d[4], self.alpha[4], self.a[4])
        H56 = self.get_DH_mat(v[5, 0], self.d[5], self.alpha[5], self.a[5])
        H02 = H01.dot(H12)
        H03 = H02.dot(H23)
        H04 = H03.dot(H34)
        H05 = H04.dot(H45)
        H06 = H05.dot(H56)
        return H06

    def get_veced_joint_poses(self, v):
        joint_pose = self.get_joint_poses(v)
        vec_joint_poses = joint_pose[:, 0:3, :].flatten()
        return vec_joint_poses

    def get_veced_ee_pose(self, v):
        joint_pose = self.get_joint_poses(v)
        vec_joint_poses = joint_pose[5, 0:3, :].flatten()
        return vec_joint_poses

    def get_veced_poses_hearder(self, prefix=""):
        element_names = ['Rxx', 'Rxy', 'Rxz', 'Tx',
                         'Ryx', 'Ryy', 'Ryz', 'Ty',
                         'Rzx', 'Rzy', 'Rzz', 'Tz']
        joint_names = ['s_p', 's_l', 'elbow', 'w1', 'w2', 'w3']
        veced_poses_hearder = []
        for joint_name in joint_names:
            for element_name in element_names:
                veced_poses_hearder.append(
                    prefix + element_name + '_' + joint_name)
        return veced_poses_hearder

    def cmpt_elbow_up_ik(self, H06):
        H06 = np.linalg.inv(self.H_base).dot(H06)
        # theta 1
        p05_homo = H06.dot(np.array([[0.0], [0.0], [-self.d[5]], [1.0]]))
        p05_x = p05_homo[0]
        p05_y = p05_homo[1]
        p05_xy_norm = np.linalg.norm(p05_homo[0:2])
        if abs(self.d[3] / p05_xy_norm) > 1:
            return None
        v0 = atan2(p05_y, p05_x) - asin(self.d[3] / p05_xy_norm)
        # theta 5
        p06_x = H06[0, 3]
        p06_y = H06[1, 3]
        cos_v4 = (-sin(v0) * p06_x + cos(v0) * p06_y - self.d[3]) / self.d[5]
        if abs(cos_v4) > 1:
            return None
        v4 = -acos(cos_v4)  # wrist down
        # theta 6
        H01 = self.get_DH_mat(v0, self.d[0], self.alpha[0], self.a[0])
        H10 = np.linalg.inv(H01)
        H16 = H10.dot(H06)
        H61 = np.linalg.inv(H16)
        H61_zx = H61[0, 2]
        H61_zy = H61[1, 2]
        v5 = atan2((H61_zy / sin(v4)), (-H61_zx / sin(v4)))
        # theta 2 and theta 3
        H45 = self.get_DH_mat(v4, self.d[4], self.alpha[4], self.a[4])
        H56 = self.get_DH_mat(v5, self.d[5], self.alpha[5], self.a[5])
        H14 = H10.dot(H06).dot(np.linalg.inv(H45.dot(H56)))
        p13_homo = H14.dot(np.array([[0.0], [self.d[3]], [0.0], [1.0]]))
        norm_p13 = np.linalg.norm(p13_homo[0:3])
        cos_v2 = (math.pow(norm_p13, 2) - math.pow(self.a[1], 2) - math.pow(
            self.a[2], 2)) / (2 * self.a[1] * self.a[2])
        if abs(cos_v2) > 1:
            return None
        v2 = acos(cos_v2)
        sin_delta = self.a[2] * sin(v2) / norm_p13
        cos_phy = p13_homo[0] / norm_p13
        if abs(cos_phy) > 1 or abs(sin_delta) > 1:
            return None
        v1 = -(asin(sin_delta) + acos(cos_phy))
        # theta 4
        H12 = self.get_DH_mat(v1, self.d[1], self.alpha[1], self.a[1])
        H23 = self.get_DH_mat(v2, self.d[2], self.alpha[2], self.a[2])
        H34 = np.linalg.inv(H12.dot(H23)).dot(H14)
        H34_xx = H34[0, 0]
        H34_xy = H34[1, 0]
        if abs(H34_xx) < 1e-10:
            return None
        v3 = atan2(H34_xy, H34_xx)
        theta = np.array([[v0], [v1], [v2], [v3], [v4], [v5]])
        return theta


def main(args):
    """
    A test script of this class
    """
    fk_ur5 = UR5Kin()
    q = np.array([1.63, -1.2, 2.5, -1.5, -0.5, 0.44])
    q = q.reshape((6, 1))
    # ee_pose = fk_ur5.get_ee_pose(q)
    X_0_E = fk_ur5.get_ee_pose(q)
    print(X_0_E)
    q_bar = fk_ur5.cmpt_elbow_up_ik(X_0_E)
    print(q_bar-q)

if __name__ == "__main__":
    main(sys.argv)
