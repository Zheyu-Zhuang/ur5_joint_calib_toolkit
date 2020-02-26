#include "common.h"

Matrix3d skew(const Vector3d& v) {
    Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return m;
}

Vector3d vex(const Matrix3d& m) {
    Vector3d v;
    v << m(2,1), m(0,2), m(1,0);
    return v;
}

Matrix3d Proj3(const Vector3d& v) {
    return Matrix3d::Identity() - v*v.transpose() / (v.squaredNorm());
}

Matrix3d NormedProj3(const Vector3d& v) {
    return Matrix3d::Identity() - v*v.transpose();
}

Matrix4d se3Exp(const Matrix4d &se3vel) {
    Vector3d u = se3vel.block<3,1>(0,3);
    Matrix3d wx = se3vel.block<3,3>(0,0);
    Vector3d w = vex(se3vel.block<3,3>(0,0));
    double th = w.norm();

    double A,B,C;
    if (abs(th) >= 1e-12) {
        A = sin(th)/th;
        B = (1-cos(th))/pow(th,2);
        C = (1-A)/pow(th,2);
    } else {
        A = 1.0;
        B = 1.0/2.0;
        C = 1.0/6.0;
    }

    Matrix3d R = Matrix3d::Identity() + A*wx + B*wx*wx;
    Matrix3d V = Matrix3d::Identity() + B*wx + C*wx*wx;

    Matrix4d expMat = Matrix4d::Identity();
    expMat.block<3,3>(0,0) = R;
    expMat.block<3,1>(0,3) = V * u;

    return expMat;
}

Matrix3d so3Log(const Matrix3d& R) {
    double theta = acos((R.trace() - 1)/2);
    double coeff = 0;
    if (abs(theta) < 1e-8) coeff = 0.5;
    else coeff = theta/(2*sin(theta));
    Matrix3d logMat = coeff * (R - R.transpose());
}

Matrix3d so3Exp(const Matrix3d &so3vel) {
    Vector3d w = vex(so3vel);
    double th = w.norm();

    double A,B;
    if (abs(th) >= 1e-12) {
        A = sin(th)/th;
        B = (1-cos(th))/pow(th,2);
    } else {
        A = 1.0;
        B = 1.0/2.0;
    }

    Matrix3d R = Matrix3d::Identity() + A*so3vel + B*so3vel*so3vel;

    return R;
}
