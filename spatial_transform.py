import copy
import math

import numpy as np
from scipy.spatial.transform.rotation import Rotation
from typing import Union

__author__ = 'Zheyu Zhuang'
__version__ = 'beta v1.0'
__maintainer__ = 'Zheyu Zhuang'
__email__ = 'zheyu.zhuang@anu.edu.au'
__status__ = 'Dev'


class SanityCheck:
    @staticmethod
    def in_SO3(matrix):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        is_orthnormal = np.linalg.norm(matrix @ matrix.T - np.eye(3)) < 1e-5
        det_is_1 = (np.linalg.det(matrix) - 1) < 1e-5
        if is_orthnormal and det_is_1:
            return True

    @staticmethod
    def in_R3(vector):
        if len(vector) == 3:
            return True
        else:
            return False

    @staticmethod
    def in_SE3(matrix):
        if matrix.shape[0] != 4 or matrix.shape[1] != 4 or matrix.ndim != 2:
            return False
        if SanityCheck.in_SO3(matrix[:3, :3]) and \
                SanityCheck.in_R3(matrix[:3, 3]):
            return True

    @staticmethod
    def in_so3(matrix):
        if matrix.shape[0] != 3 or matrix.shape[1] != 3 or matrix.ndim != 2:
            return False
        if np.linalg.norm(matrix + matrix.T) < 1e-8:
            return True

    @staticmethod
    def can_mul(A, B):
        return A.shape[1] == B.shape[0]

    @staticmethod
    def can_add(A, B):
        return np.all(A.shape == B.shape)


class SpatialTransform:
    """
    This Class Defines a template for spatial transformation child-classes.
    SE3 and SO3 are closed under matrix multiplication, but not addition.
    R3 is closed under vector addiction (hence, scalar multiplication).
    The exponential maps se3, so3 are closed under addition, but not
    multiplication
    """
    __array_priority__ = 1000

    @classmethod
    def from_array(cls, input_array):
        pass

    def __init__(self, input_array):
        self.value = input_array

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, input_array):
        self.__value = input_array

    def __str__(self):
        temp = np.around(self.value, decimals=5)
        return temp.__str__()

    def __repr__(self):
        temp = np.around(self.value, decimals=5)
        return f"{self.__class__.__name__}\n{temp.__str__()}"

    def to_array(self):
        return self.value


class CloseUnderMul(SpatialTransform):

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            if SanityCheck.can_mul(self.value, other):
                return self.value @ other
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        elif isinstance(other, self.__class__):
            if SanityCheck.can_mul(self.value, other.value):
                return self.__class__.from_array(self.value @ other.value)
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            if SanityCheck.can_mul(self.value, other):
                return other @ self.value
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        elif isinstance(other, self.__class__):
            if SanityCheck.can_mul(self.value, other.value):
                return self.__class__.from_array(other.value @ self.value)
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            other = other.value
        elif isinstance(other, np.ndarray):
            pass
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

        if SanityCheck.can_add(self.value, other):
            return self.value - other
        else:
            raise Exception('Broadcast is disabled, dimension mismatch')

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other.value
        elif isinstance(other, np.ndarray):
            pass
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

        if SanityCheck.can_add(self.value, other):
            return self.value + other
        else:
            raise Exception('Broadcast is disabled, dimension mismatch')

    __radd__ = __add__

    def __mul__(self, other):
        custom_type = Union[int, float]
        if isinstance(other, custom_type.__args__):
            return self.value * other
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

    __rmul__ = __mul__


class CloseUnderAdd(SpatialTransform):

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            other = other.value
        elif isinstance(other, np.ndarray):
            pass
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')
        if SanityCheck.can_mul(self.value, other):
            return self.value @ other
        else:
            raise Exception('Broadcast is disabled, dimension mismatch')

    def __rmatmul__(self, other):
        if isinstance(other, self.__class__):
            other = other.value
        elif isinstance(other, np.ndarray):
            pass
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

        if SanityCheck.can_mul(self.value, other):
            return other @ self.value
        else:
            raise Exception('Broadcast is disabled, dimension mismatch')

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            if SanityCheck.can_mul(self.value, other):
                return self.value - other
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        elif isinstance(other, self.__class__):
            if SanityCheck.can_mul(self.value, other.value):
                return self.__class__.from_array(self.value - other.value)
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __neg__(self):
        return self.__class__.from_array(-self.value)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            if SanityCheck.can_mul(self.value, other):
                return self.value - other
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        elif isinstance(other, self.__class__):
            if SanityCheck.can_mul(self.value, other.value):
                return self.__class__.from_array(self.value + other.value)
            else:
                raise Exception('Broadcast is disabled, dimension mismatch')
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not supported')

    __radd__ = __add__

    def __mul__(self, other):
        custom_type = Union[int, float]
        if isinstance(other, custom_type.__args__):
            return self.__class__.from_array(self.value * other)
        else:
            raise Exception(
                f'Operation with {other.__class__.__name__} is not surpported')

    __rmul__ = __mul__


class R3(CloseUnderAdd):
    @classmethod
    def random(cls, trans_min=(0.2, -0.3, 0.0),
               trans_max=(0.6, 0.3, 0.1)):
        rand_trans = np.random.uniform(list(trans_min), list(trans_max))
        return R3.from_array(rand_trans)

    @classmethod
    def from_array(cls, input_array=None):
        if isinstance(input_array, cls):
            return copy.copy(input_array)
        else:
            return R3(input_array)

    def __init__(self, input_array=None):
        if input_array is None:
            input_array = np.zeros(3)
        input_array = np.asarray(input_array).reshape(3)
        if SanityCheck.in_R3(input_array):
            super().__init__(input_array)
        else:
            raise Exception('Input array is not in R3')

    def transform(self, SE3_):
        SE3_temp = SE3_ @ (self.to_SE3())
        return SE3_temp.t

    def to_SE3(self):
        out = SE3()
        out.t = self
        return out


class SO3(CloseUnderMul):

    @classmethod
    def from_array(cls, input_array):
        if isinstance(input_array, cls):
            return copy.copy(input_array)
        else:
            return cls(input_array)

    @classmethod
    def from_rotvec(cls, rotvec):
        rotation = Rotation.from_rotvec(rotvec)
        return cls(rotation.as_matrix())

    @classmethod
    def from_quat(cls, quat):
        rotation = Rotation.from_quat(quat)
        return cls(rotation.as_matrix())

    @classmethod
    def random(cls):
        return cls(Rotation.random().as_matrix())

    @classmethod
    def from_basic_rotation(cls, theta: float, axis: str):
        c = math.cos(theta)
        s = math.sin(theta)
        if axis == 'x':
            return SO3.from_array(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))
        elif axis == 'y':
            return SO3.from_array(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
        elif axis == 'z':
            return SO3.from_array(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
        else:
            raise Exception('choose axis from x, y, z')

    @classmethod
    def get_basic_rotation_generators(cls, axis: str):
        if axis == 'x':
            M1 = np.zeros([3, 3]);
            M1[1, 1] = 1;
            M1[2, 2] = 1
            M2 = np.zeros([3, 3]);
            M2[1, 2] = -1;
            M2[2, 1] = 1
            M3 = np.zeros([3, 3]);
            M3[0, 0] = 1
            return M1, M2, M3
        elif axis == 'y':
            M1 = np.zeros([3, 3]);
            M1[0, 0] = 1;
            M1[2, 2] = 1
            M2 = np.zeros([3, 3]);
            M2[0, 2] = 1;
            M2[2, 0] = -1
            M3 = np.zeros([3, 3]);
            M3[1, 1] = 1
            return M1, M2, M3
        elif axis == 'z':
            M1 = np.zeros([3, 3]);
            M1[0, 0] = 1;
            M1[1, 1] = 1
            M2 = np.zeros([3, 3]);
            M2[0, 1] = -1;
            M2[1, 0] = 1
            M3 = np.zeros([3, 3]);
            M3[2, 2] = 1
            return M1, M2, M3
        else:
            raise Exception(f'{axis} is an element of [x, y, z]')

    def __init__(self, input_array=None):
        if input_array is None:
            input_array = np.eye(3)
        input_array = np.asarray(input_array).reshape(3, 3)
        if SanityCheck.in_SO3(input_array):
            super().__init__(input_array)
        else:
            raise Exception('Input array is not in SO3')

    def rotate(self, rot_mat, refer_to='parent'):
        other = SO3.from_array(rot_mat)
        if refer_to == 'parent':
            return other @ self
        elif refer_to == 'self':
            return self @ other
        else:
            raise Exception('reference frame is not defined')

    def log(self):
        _SO3 = copy.copy(self.value)
        try:
            theta = math.acos((np.matrix.trace(_SO3) - 1) / 2)
        except:
            theta = 0.0
        if abs(theta) < 1e-8:
            A = 0.5
        else:
            A = theta / (2 * math.sin(theta))
        return so3.from_array(A * (_SO3 - np.matrix.transpose(_SO3)))

    def to_SE3(self):
        out = SE3()
        out.R = self
        return out

    def inv(self):
        return SO3.from_array(self.value.T)

    def T(self):
        return self.inv()


class SE3(CloseUnderMul):

    @classmethod
    def from_array(cls, matrix):
        if isinstance(matrix, cls):
            return copy.copy(matrix)
        elif matrix.shape[0] != 4 or matrix.shape[1] != 4 or matrix.ndim != 2:
            raise Exception('input dimension is not 4x4')
        else:
            return cls(matrix[:3, :3], matrix[:3, 3])

    @classmethod
    def random(cls):
        out = SE3()
        out.R = SO3.random()
        out.t = R3.random()
        return out

    def __init__(self, rot_mat=None, position=None):
        self.t = R3.from_array(position)
        self.R = SO3.from_array(rot_mat)

    @property
    def value(self):
        out = np.eye(4)
        out[0:3, 0:3], out[0:3, 3] = self.R.value, self.t.value
        return out

    @property
    def R(self):
        return self.__R

    @R.setter
    def R(self, input_obj: SO3):
        if isinstance(input_obj, SO3):
            self.__R = input_obj
        elif isinstance(input_obj, np.ndarray):
            self.__R = SO3.from_array(input_obj)
        else:
            raise Exception(f'{input_obj.__class__} is not supported')

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, input_obj: R3):
        if isinstance(input_obj, R3):
            self.__t = input_obj
        elif isinstance(input_obj, np.ndarray):
            self.__t = R3.from_array(input_obj)
        else:
            raise Exception(f'{input_obj.__class__} is not supported')

    def rotate(self, rot_mat, refer_to='parent'):
        other = SO3.from_array(rot_mat).to_SE3()
        if refer_to == 'parent':
            return other @ self
        elif refer_to == 'self':
            return self @ other
        else:
            raise Exception('reference frame is not defined')

    def translate(self, trans_vec, refer_to='parent'):
        other = R3.from_array(trans_vec).to_SE3()
        if refer_to == 'parent':
            return other @ self
        elif refer_to == 'self':
            return self @ other
        else:
            raise Exception('reference frame is not defined')

    def log(self):
        omega_x_np = self.R.log().value
        t_np = self.t.value
        omega_np = self.R.log().as_vector().reshape(3, 1)
        theta = np.linalg.norm(omega_np)
        A = -0.5
        B = 1.0 / 12.0
        if abs(theta) >= 1e-8:
            B = (1 - (theta * math.sin(theta) / (
                    2 * (1 - math.cos(theta))))) / (theta ** 2)
        V_inv = np.eye(3) + A * omega_x_np + B * omega_x_np @ omega_x_np
        v = V_inv @ (t_np.reshape(3, 1))
        return se3(v, omega_x_np)

    def T(self):
        return copy.copy(self.value.T)

    def inv(self):
        return SE3.from_array(np.linalg.inv(self.value))


# Exponential Map of S0(3) and SE(3)
class so3(CloseUnderAdd):

    @classmethod
    def random(cls, omega_min=(0.2, -0.3, 0.0),
               omega_max=(0.6, 0.3, 0.1)):
        rand_trans = np.random.uniform(list(omega_min), list(omega_max))
        return so3.from_vector(rand_trans)

    @classmethod
    def from_vector(cls, input_vector):
        '''
        convert the minimum parametrisation of angular velocity to so3 object
        @param input_vector: [omega1, omega2, omega3]
        @return:
        '''
        omega = np.asarray(input_vector).reshape(3)
        if SanityCheck.in_R3(omega):
            _so3 = np.zeros((3, 3))
            _so3[1, 0] = omega[2]
            _so3[2, 0] = -omega[1]
            _so3[2, 1] = omega[0]
            _so3 -= _so3.T
            return cls(_so3)
        else:
            raise Exception('input vector is not in R3')

    @classmethod
    def from_array(cls, input_array):
        if isinstance(input_array, cls):
            return copy.copy(input_array)
        else:
            return cls(input_array)

    @classmethod
    def from_nearest_projection(cls, input_array):
        if input_array.ndim != 2 or input_array.shape[0] != 3 or \
                input_array.shape[1] != 3:
            raise Exception('Input input_array is not in R^{3x3}')
        else:
            return so3.from_array(0.5 * (input_array - input_array.T))

    def __init__(self, input_array=None):
        if input_array is None:
            input_array = np.zeros((3, 3))
        input_array = np.asarray(input_array).reshape(3, 3)
        if SanityCheck.in_so3(input_array):
            super().__init__(input_array)
        else:
            raise Exception('Input array is not in so(3)')

    def as_vector(self):
        """
        @return: convert so3 object to minimum parametrisation
        """
        return np.array([self.value[2, 1], self.value[0, 2], self.value[1, 0]])

    def exp(self):
        omega_np = self.as_vector()
        omega_x_np = copy.copy(self.value)
        th = np.linalg.norm(omega_np)
        A = 1
        B = 0.5
        if abs(th) >= 1e-8:
            A = math.sin(th) / th
            B = (1 - math.cos(th)) / pow(th, 2)
        return SO3.from_array(np.eye(3) + A * omega_x_np
                              + B * (omega_x_np.value @ omega_x_np))


class se3(CloseUnderAdd):
    @classmethod
    def from_array(cls, input_array):
        if isinstance(input_array, cls):
            return copy.copy(input_array)
        elif input_array.shape[0] != 4 or input_array.shape[1] != 4 \
                or input_array.ndim != 2:
            raise Exception('input dimension is not 4x4')
        else:
            return cls(R3.from_array(input_array[:3, 3]),
                       so3.from_array(input_array[:3, :3]))

    @classmethod
    def from_vector(cls, input_vector):
        input_vector = input_vector.reshape(6)
        return se3(R3.from_array(input_vector[:3]),
                   so3.from_vector(input_vector[3:]))

    @classmethod
    def from_nearest_projection(cls, input_array):
        if input_array.ndim != 2 or input_array.shape[0] != 4 or \
                input_array.shape[1] != 4:
            raise Exception('Input input_array is not in R^{4x4}')
        se3_approx = np.zeros((4, 4))
        se3_approx[0:3, 0:3] = so3.from_nearest_projection(
            input_array[0:3, 0:3]).value
        se3_approx[0:3, 3] = input_array[0:3, 3]
        return se3.from_array(se3_approx)

    def __init__(self, v=None, omega_x=None):
        self.v = R3.from_array(v)
        self.omega_x = so3.from_array(omega_x)

    @property
    def value(self):
        out = np.zeros((4, 4))
        out[0:3, 0:3], out[0:3, 3] = self.omega_x.value, self.v.value
        return out

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, input_obj: R3):
        if isinstance(input_obj, R3):
            self.__v = input_obj
        elif isinstance(input_obj, np.ndarray):
            self.__v = R3.from_array(input_obj)
        else:
            raise Exception(f'{input_obj.__class__} is not supported')

    @property
    def omega_x(self):
        return self.__omega_x

    @omega_x.setter
    def omega_x(self, input_obj: so3):
        if isinstance(input_obj, so3):
            self.__omega_x = input_obj
        elif isinstance(input_obj, np.ndarray):
            self.__omega_x = so3.from_array(input_obj)
        else:
            raise Exception(f'{input_obj.__class__} is not supported')

    def as_vector(self):
        return np.array([self.value[0, 3], self.value[1, 3], self.value[2, 3],
                         self.value[2, 1], self.value[0, 2], self.value[1, 0]])

    def exp(self):
        omega_np = self.omega_x.as_vector().reshape(3, 1)
        omega_x_np = copy.copy(self.omega_x.value)
        v_np = copy.copy(self.v.value)
        theta = np.linalg.norm(omega_np)
        A = 0.5
        B = 1.0 / 6.0
        if abs(theta) >= 1e-8:
            A = (1 - math.cos(theta)) / (theta ** 2)
            B = (theta - math.sin(theta)) / (theta ** 3)
        V = np.eye(3) + A * omega_x_np + B * (omega_x_np @ omega_x_np)
        exp_se3 = np.eye(4)
        exp_se3[:3, :3] = self.omega_x.exp().value
        exp_se3[:3, 3] = V.dot(v_np).reshape(3)
        return SE3.from_array(exp_se3)
