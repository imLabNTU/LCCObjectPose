//
//  DataStructure.h
//  PCM
//
//  Created by Lau Yo-Chung on 2021/2/20.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#ifndef DataStructure_h
#define DataStructure_h

#include <fstream>

#include <eigen3/Eigen/Dense>
#include "opencv2/core.hpp"

struct BasicPose
{
    BasicPose(Eigen::Matrix3d r_ = Eigen::Matrix3d::Identity(), Eigen::Vector3d t_ = Eigen::Vector3d())
    {
        r = r_;
        t = t_;
        isActive = true;
    };

    Eigen::Matrix3d r;
    Eigen::Vector3d t;

    bool isActive;
};

struct FrameInfo
{
    FrameInfo(): rate(30), period(1000.0 / (double) rate) {
        toNextTime = duration = period;
    }

    cv::Mat frame;

    std::string basePath;
    std::string baseBoxPath;
    std::string fileName;

    int rate;
    long index;

    double toNextTime;              // ms
    double duration;                // ms
    double period;                  // 1000 / rate (ms)
    double time_s;
    double time_ms;                 // in s
    double time;                    // in s
    double _time;                   // in s
    double timeInReading;           // in s

    Eigen::Vector3d _p;
    Eigen::Vector3d p;
    Eigen::Vector3d e;
    Eigen::Matrix3d r;
    Eigen::Vector3d pGT;
    Eigen::Vector3d eGT;
    Eigen::Matrix3d rGT;
    Eigen::Vector3d vGT;
    Eigen::Vector3d wGT;

    bool status;
};

struct ObjectInfo
{
    ObjectInfo() {
        r = Eigen::Matrix3d::Identity();
        t = Eigen::Vector3d();
        s = Eigen::Vector3d(1, 1, 1);
    };

    void transform() {
        boxesRT.clear();

        for (int i = 0; i < boxes.size(); i++)
            boxesRT.push_back(r * boxes[i] + t);
    };
    
    std::vector<Eigen::Vector3d> boxes;
    std::vector<Eigen::Vector3d> boxesRT;

    Eigen::Matrix3d r;
    Eigen::Vector3d t;
    Eigen::Vector3d s;
};

#endif /* DataStructure_h */
