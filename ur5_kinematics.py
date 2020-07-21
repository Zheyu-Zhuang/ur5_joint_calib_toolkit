import sys
from math import sin, cos, pi, acos, asin, atan2

from .spatial_transform import *


class UR5Kinematics:
    """
    This class defines functions related to the forward kinematics of UR5 arm
    The base frame is defined w.r.t the manipulator's base frame
    """

    def __init__(self, grasping_offset=0.0):
        # define UR5_Params
        self.d_ee = 0.224 + grasping_offset  # end-effector offset
        self.a = np.array([0, 0.42500, 0.39225, 0, 0, 0])
        self.d = np.array(
            [0.089159, 0, 0, 0.10915, 0.09465, 0.0823 + self.d_ee])
        self.alpha = np.array([-pi / 2, 0, 0, -pi / 2, pi / 2, 0])
        # self.H_base = np.array([[0.707, 0.707, 0, 0],
        #                         [-0.707, 0.707, 0, 0],
        #                         [0, 0, 1, 0.931],
        #                         [0, 0, 0, 1]])
        self.H_base = SE3()

    def get_DH(self, theta: float, joint_id: int) -> SE3:
        d, alpha, a = self.d[joint_id], self.alpha[joint_id], self.a[joint_id]
        H = np.array([[cos(theta), -sin(theta) * cos(alpha),
                       sin(theta) * sin(alpha), a * cos(theta)],
                      [sin(theta), cos(theta) * cos(alpha),
                       -cos(theta) * sin(alpha), a * sin(theta)],
                      [0.0, sin(alpha), cos(alpha), d],
                      [0.0, 0.0, 0.0, 1.0]])
        return SE3.from_array(H)

    def get_pd_DH(self, theta: float, joint_id: int) -> np.ndarray:
        d, alpha, a = self.d[joint_id], self.alpha[joint_id], self.a[joint_id]
        pd_H = np.array([[-sin(theta), -cos(theta) * cos(alpha),
                          cos(theta) * sin(alpha), -a * sin(theta)],
                         [cos(theta), -sin(theta) * cos(alpha),
                          sin(theta) * sin(alpha), a * cos(theta)],
                         [0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])
        return pd_H

    def get_pd_X_H(self, theta: np.ndarray, joint_id: int) -> np.ndarray:
        all_DH = []
        for i in range(6):
            all_DH.append(self.get_DH(theta[i, 0], joint_id=i))
        all_DH[joint_id] = self.get_pd_DH(theta[joint_id, 0], joint_id=joint_id)
        out = SE3()
        for joint_DH in all_DH:
            out = out @ joint_DH
        return out

    def get_all_joint_poses(self, theta: np.ndarray) -> list:
        all_joint_poses = [self.H_base @ self.get_DH(theta[0, 0],
                                                         joint_id=0)]
        for i in range(1, 6):
            all_joint_poses.append(all_joint_poses[i - 1] @
                                   self.get_DH(theta[i, 0], joint_id=i))
        return all_joint_poses

    def get_ee_pose(self, theta: np.ndarray):
        return self.get_all_joint_poses(theta)[-1]

    def cmpt_jacobian(self, theta, ref='world'):
        joint_poses = self.get_all_joint_poses(theta)
        J = np.zeros((6, 6))
        p = joint_poses[5].value[0:3, 3]
        for i in range(0, 6):
            if i == 0:
                z_i_1 = np.array([0, 0, 1])
                p_i_1 = np.zeros(3)
            else:
                z_i_1 = joint_poses[i - 1].value[0:3, 2]
                p_i_1 = joint_poses[i - 1].value[0:3, 3]
            J[0:3, i] = np.cross(z_i_1, (p - p_i_1))
            J[3:6, i] = z_i_1
        if ref == 'ee':
            X_H = joint_poses[5]
            R_ee = X_H.R
            R_compound = np.zeros((6, 6))
            R_compound[0:3, 0:3], R_compound[3:6, 3:6] = R_ee.value, R_ee.value
            J = R_compound.transpose().dot(J)
        elif ref == 'world':
            pass
        else:
            print('Invalid Reference frame, returning J in inertia frame')
        return J

    def cmpt_jacobian_dumb(self, theta):
        X = self.get_ee_pose(theta)
        jacobian = np.zeros((6, 6))
        for joint_id in range(6):
            pd_X = self.get_pd_X_H(theta, joint_id)
            jacobian[0:3, joint_id] = pd_X[0:3, 3]
            omega_x = so3.from_array((pd_X @ X.inv())[0:3, 0:3])
            jacobian[3:6, joint_id] = omega_x.as_vector()
        return jacobian

    def cmpt_elbow_up_ik(self, H06: SE3) -> Union[None, np.ndarray]:
        H06 = self.H_base.inv() @ H06
        # theta 1
        p05_homo = H06 @ np.array([[0.0], [0.0], [-self.d[5]], [1.0]])
        p05_x = p05_homo[0]
        p05_y = p05_homo[1]
        p05_xy_norm = np.linalg.norm(p05_homo[0:2])
        if abs(self.d[3] / p05_xy_norm) > 1:
            return None
        v0 = atan2(p05_y, p05_x) - asin(self.d[3] / p05_xy_norm)
        # theta 5
        p06_x = H06.value[0, 3]
        p06_y = H06.value[1, 3]
        cos_v4 = (-sin(v0) * p06_x + cos(v0) * p06_y - self.d[3]) / self.d[5]
        if abs(cos_v4) > 1:
            return None
        v4 = -acos(cos_v4)  # wrist down
        # theta 6
        H01 = self.get_DH(v0, joint_id=0)
        H10 = H01.inv()
        H16 = H10 @ H06
        H61 = H16.inv()
        H61_zx = H61.value[0, 2]
        H61_zy = H61.value[1, 2]
        v5 = atan2((H61_zy / sin(v4)), (-H61_zx / sin(v4)))
        # theta 2 and theta 3
        H45 = self.get_DH(v4, joint_id=4)
        H56 = self.get_DH(v5, joint_id=5)
        H46 = H45 @ H56
        H14 = H10 @ H06 @ H46.inv()
        p13_homo = H14 @ np.array([[0.0], [self.d[3]], [0.0], [1.0]])
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
        H12 = self.get_DH(v1, joint_id=1)
        H23 = self.get_DH(v2, joint_id=2)
        H13 = H12 @ H23
        H34 = H13.inv() @ H14
        H34_xx = H34.value[0, 0]
        H34_xy = H34.value[1, 0]
        if abs(H34_xx) < 1e-10:
            return None
        v3 = atan2(H34_xy, H34_xx)
        theta = np.array([[v0], [v1], [v2], [v3], [v4], [v5]])
        return theta


def main(args):
    """
    A test script of this class
    """
    fk_ur5 = UR5Kinematics()
    q = np.array([2.35619, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708])
    q = q.reshape((6, 1))
    ee_pose = fk_ur5.get_ee_pose(q)
    jacobian = fk_ur5.cmpt_jacobian(q)
    jacobian_dumb = fk_ur5.cmpt_jacobian_dumb(q)
    print(ee_pose)
    print(np.around(jacobian, decimals=3))
    print(np.around(jacobian_dumb, decimals=3))
    print(np.around(jacobian, decimals=3) - np.around(jacobian_dumb, decimals=3))


if __name__ == "__main__":
    main(sys.argv)
