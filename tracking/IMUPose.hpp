//
//  IMUPose.hpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/3/13.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#ifndef IMUPose_hpp
#define IMUPose_hpp

#include <iostream>

#include <eigen3/Eigen/Dense>

class IMUPose
{
public:
    IMUPose();
    ~IMUPose();

    static Eigen::Matrix4d getOmegaMatrix(Eigen::Vector3d w);
    static IMUPose integral(Eigen::Vector3d w, Eigen::Vector3d a, Eigen::Vector3d v, Eigen::Vector4d q, Eigen::Vector3d g);

    /*
     w: the angular velocity of body
     a: the acceleration of body
     t: delta time
    */
    void clearAccumulation();
    void saveAccumulation(Eigen::Vector3d w, Eigen::Vector3d a, double t, double time);
    void clearLost();
    void saveLost(Eigen::Vector3d w, Eigen::Vector3d a, double t, double time);

    void update(bool isRK4, bool is4Lost = false);
    void update(Eigen::Vector3d w, Eigen::Vector3d a, double t);
    void rk4(Eigen::Vector3d w, Eigen::Vector3d a, double t);

    Eigen::Vector4d q;  // w, x, y, z
    Eigen::Matrix3d r;  // self rotation matrix from body to world
    Eigen::Vector3d v;  // self velocity in world
    Eigen::Vector3d p;  // self postion in world

    std::vector<Eigen::Vector3d> ws;
    std::vector<Eigen::Vector3d> as;
    std::vector<double> ts;
    std::vector<double> whens;

    std::vector<Eigen::Vector3d> wsLost;
    std::vector<Eigen::Vector3d> asLost;
    std::vector<double> tsLost;
    std::vector<double> whensLost;

    Eigen::Matrix3d rAC;
    Eigen::Vector3d vAC;
    Eigen::Vector3d pAC;

    bool isAdjustW;
    Eigen::Vector3d bias_a;
    Eigen::Vector3d bias_w;

    Eigen::Vector3d _w;
    Eigen::Vector3d _a;
    Eigen::Vector3d _v;
    Eigen::Vector3d _p;

    Eigen::Vector3d w;
    Eigen::Vector3d a;
    Eigen::Vector3d g;  // the acceleration of gravity, default = -9.8 (m / s^2)

    int rate;           // updating frequency, default = 200
    double toNextTime;  // ms
    double duration;    // ms
    double period;      // 1000 / rate (ms)
    double time;
    double timeRead;

    bool status;

private:
    void updateP(double t);                    // t: delta time
    void updateV(Eigen::Vector3d a, double t); // a: the acceleration of body; t: delta time
    void updateR(Eigen::Vector3d w, double t); // w: the angular velocity of body; t: delta time
};

#endif /* IMUPose_hpp */
