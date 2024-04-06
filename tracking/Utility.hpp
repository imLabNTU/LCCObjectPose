//
//  utility.hpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/3/13.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#ifndef utility_hpp
#define utility_hpp

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

enum RotSeq
{
    zyz,
    zxy,
    zxz,
    yxz,
    yxy,
    yzx,
    yzy,
    xyz,
    xyx,
    xzy,
    xzx,
    zyx
};

std::vector<std::string> getLineSub(std::string line, char sepSymbol);
double getElapsedTime(double start_tic);

Eigen::Matrix3d getSkewMatrix3(Eigen::Vector3d w);
Eigen::VectorXd interpolateVectorXd(Eigen::VectorXd v1, Eigen::VectorXd v2, double step, double steps);

void project3DPoints(cv::Mat rMatrix, cv::Mat tVec, cv::Mat k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos);
void project3DPoints(cv::Mat pose, cv::Mat k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos);
void project3DPoints(Eigen::Matrix3d r, Eigen::Vector3d t, Eigen::Matrix3d k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos);
void project3DPoints(std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos, cv::Mat eulers, cv::Mat tVec, cv::Mat k);

// RzRyRx
Eigen::Matrix3d eulers2Matrix(double yawZ, double pitchY, double rollX);  // in radius
void eulers2Mat(double yawZ, double pitchY, double rollX, cv::Mat &rRot); // in radius

Eigen::Vector3d quaternion2Eulers(Eigen::Quaterniond qt, RotSeq rotSeq = zyx);
Eigen::Quaterniond eulers2Quaterniond(Eigen::Vector3d e);                     // RzRyRx
Eigen::Quaterniond eulers2Quaterniond(double yaw, double pitch, double roll); // yaw (Z), pitch (Y), roll (X), by RzRyRx

Eigen::Matrix3d quaternion2Matrix(Eigen::Quaterniond q);

int getRandomUnifrom(int lower, int upper);
double getRandomNormal(double mean, double sigma, double maximum = -1);

std::string itosWith0(long number, int numberOf0 = 8);

#endif /* utility_hpp */
