//
//  IMUPose.cpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/3/13.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#include "Utility.hpp"
#include "IMUPose.hpp"

using namespace std;
using namespace Eigen;

IMUPose::IMUPose()
{
    r = rAC = Matrix3d::Identity();
    q = Vector4d(1, 0, 0, 0);
    v = vAC = Vector3d(0, 0, 0);
    p = pAC = Vector3d(0, 0, 0);
    g = Vector3d(0, 0, -9.8);
//    g = Vector3d(9.8, 0, 0);
    a = Vector3d(0, 0, 0);
    w = Vector3d(0, 0, 0);
    _a = Vector3d(0, 0, 0);
    _w = Vector3d(0, 0, 0);
    _v = Vector3d(0, 0, 0);
    _p = Vector3d(0, 0, 0);

    isAdjustW = true;
    bias_a = bias_w = Vector3d(0, 0, 0);

    rate = 200;
    period = 1000.0 / ((double) rate);
    time = timeRead = 0;

    status = true;
}

IMUPose::~IMUPose()
{
}

Matrix4d IMUPose::getOmegaMatrix(Vector3d w)
{
    Matrix3d sk = getSkewMatrix3(w);
    Matrix4d m;
    m.block(1, 1, 3, 3) = -sk;
    m.block(0, 1, 1, 3) = -(w.transpose());
    m.block(1, 0, 3, 1) = w;
    return m;
}

IMUPose IMUPose::integral(Vector3d w, Vector3d a, Vector3d v, Vector4d q, Vector3d g)
{
    IMUPose imu;
    Vector4d qn(q);
    qn = qn / qn.norm();
    Quaterniond qt = Quaterniond(qn(0), qn(1), qn(2), qn(3));
    Matrix3d rMatix = qt.toRotationMatrix();
    Matrix4d omega = getOmegaMatrix(w);

    imu.q = 0.5 * omega * qn;
    imu.v = rMatix * a + g;
    imu.p = v;
    imu.g = g;

    return imu;
}

void IMUPose::rk4(Vector3d w, Vector3d a, double t)
{
    Vector3d ak1 = _a;
    Vector3d wk1 = _w;

    Vector3d ak23 = interpolateVectorXd(_a, a, 0.5 * t, t);
    Vector3d wk23 = interpolateVectorXd(_w, w, 0.5 * t, t);

    Vector3d ak4 = a;
    Vector3d wk4 = w;

    IMUPose state_der1 = IMUPose::integral(wk1, ak1, v, q, g);
    IMUPose state_der2 = IMUPose::integral(wk23, ak23, v + 0.5 * t * state_der1.v, q + 0.5 * t * state_der1.q, g);
    IMUPose state_der3 = IMUPose::integral(wk23, ak23, v + 0.5 * t * state_der2.v, q + 0.5 * t * state_der2.q, g);
    IMUPose state_der4 = IMUPose::integral(wk4, ak4, v + t * state_der3.v, q + t * state_der3.q, g);

    _a = a;
    _w = w;
    _v = v;
    _p = p;

    q += (t * (state_der1.q + 2 * state_der2.q + 2 * state_der3.q + state_der4.q) / 6);
    v += (t * (state_der1.v + 2 * state_der2.v + 2 * state_der3.v + state_der4.v) / 6);
    p += (t * (state_der1.p + 2 * state_der2.p + 2 * state_der3.p + state_der4.p) / 6);
}

#define _RatioLast 0
#define _RatioNow 1

void IMUPose::clearAccumulation()
{
    rAC = Matrix3d::Identity();
    pAC = vAC = Vector3d(0, 0, 0);

    ws.clear();
    as.clear();
    ts.clear();
    whens.clear();
}

void IMUPose::saveAccumulation(Eigen::Vector3d w, Eigen::Vector3d a, double t, double time)
{
    ws.push_back(w);
    as.push_back(a);
    ts.push_back(t);
    whens.push_back(time);
}

void IMUPose::clearLost()
{
    wsLost.clear();
    asLost.clear();
    tsLost.clear();
    whensLost.clear();
}

void IMUPose::saveLost(Eigen::Vector3d w, Eigen::Vector3d a, double t, double time)
{
    wsLost.push_back(w);
    asLost.push_back(a);
    tsLost.push_back(t);
    whensLost.push_back(time);
}

void IMUPose::update(bool isRK4, bool is4Lost)
{
    auto *wsp = is4Lost? &wsLost : &ws;
    auto *asp = is4Lost? &asLost : &as;
    auto *tsp = is4Lost? &tsLost : &ts;
    auto *whensp = is4Lost? &whensLost : &whens;

    if (!tsp->empty())
    {
        for (int i = 0; i < tsp->size(); i++)
        {
            if (time >= (*whensp)[i])
                continue;
            else
                time = (*whensp)[i];

            if (isRK4)
                rk4((*wsp)[i], (*asp)[i], (*tsp)[i]);
            else
                update((*wsp)[i], (*asp)[i], (*tsp)[i]);
        }

        if (is4Lost)
            clearLost();
        else
            clearAccumulation();
    }
}

void IMUPose::update(Vector3d w, Vector3d a, double t)
{
    if (t > 0)
    {
        updateR(w, t);
        updateV(a, t);
//        updateP(t);
    }
}

void IMUPose::updateR(Vector3d w, double t)
{
    if (t > 0)
    {
        if (isAdjustW)
            w += bias_w;
        
        Vector3d c = (_RatioLast * _w + _RatioNow * w) / (_RatioLast + _RatioNow);
        _w = w;
        c *= t;

        double d = c.norm();
        if (d != 0)
        {
            Matrix3d I = Matrix3d::Identity();
            Matrix3d B = getSkewMatrix3(c);
            Matrix3d delta = (I + ((sin(d) / d) * B) + (((1 - cos(d)) / (d * d)) * (B * B)));

            r *= delta;
            rAC *= delta;
        }
    }
}

void IMUPose::updateV(Vector3d a, double t)
{
    if (t > 0)
    {
        Vector3d c = r * ((_RatioLast * _a + _RatioNow * a) / (_RatioLast + _RatioNow));
//        cout << "a: " << a << endl;
//        cout << "c: " << c << endl;
//        cout << "c + g + bias_a: " << c + g + bias_a << endl;
        Vector3d delta = ((c + g + bias_a) * t);

        {
            Vector3d deltaP = ((_RatioLast * _v + _RatioNow * v) * t / (_RatioLast + _RatioNow));
            Vector3d deltaPbyA = ((_RatioLast * delta + _RatioNow * delta) * 0.5 * t / (_RatioLast + _RatioNow));
            _p = p;

            p += (deltaP + deltaPbyA);
            pAC += (deltaP + deltaPbyA);
        }

        _a = a;
        _v = v;

        v += delta;
        vAC += delta;
    }
}

void IMUPose::updateP(double t)
{
    if (t > 0)
    {
        Vector3d delta = ((_RatioLast * _v + _RatioNow * v) * t / (_RatioLast + _RatioNow));
        _p = p;

        p += delta;
        pAC += delta;
    }
}
