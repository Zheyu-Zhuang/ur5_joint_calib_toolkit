import numpy as np
import math


def SO3_log(_SO3):
    try:
        theta = math.acos((np.matrix.trace(_SO3) - 1) / 2)
    except:
        theta = 0.0
    if abs(theta) < 1e-8:
        coeff = 0.5
    else:
        coeff = theta / (2 * math.sin(theta))
    return coeff * (_SO3 - np.matrix.transpose(_SO3))


def so3_to_vec(_so3):
    return np.array([_so3[2, 1], _so3[0, 2], _so3[1, 0]]).reshape(3, 1)


def so3_exp(_so3):
    w = so3_to_vec(_so3).reshape(3)
    th = np.linalg.norm(w)
    A = 1
    B = 0.5
    if abs(th) >= 1e-8:
        A = math.sin(th) / th
        B = (1 - math.cos(th)) / pow(th, 2)
    return np.eye(3) + A * _so3 + B * _so3.dot(_so3)


def SO3_avg(SO3_stack, err_th=0.01):
    # implement Karcher mean / geodesic L2-mean on SO(3)
    n = SO3_stack.shape[0]
    out = SO3_stack[0, :, :]
    err = 100
    while err > err_th:
        _so3_err_sum = np.zeros([3, 3])
        for i in range(n):
            _so3_err_sum += SO3_log(
                np.transpose(out).dot(SO3_stack[i, :, :]))
        out = out.dot(so3_exp(_so3_err_sum / n))
        err = np.linalg.norm(_so3_err_sum)
        # print(err)
    return out


def SE3_avg(SE3_stack):
    out = np.eye(4)
    out[:, 3] = np.mean(SE3_stack[:, :, 3], axis=0)
    out[:3, :3] = SO3_avg(SE3_stack[:, :3, :3])
    return out
