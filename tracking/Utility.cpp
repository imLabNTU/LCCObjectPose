//
//  utility.cpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/3/13.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#include <sstream>
#include <iomanip>
#include <random>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Utility.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

struct uContainer
{
    uContainer() : w(0.0), x(0.0), y(0.0), z(0.0) {}
    uContainer(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    double w;
    double x;
    double y;
    double z;
};

std::vector<std::string> getLineSub(std::string line, char sepSymbol)
{
    vector<string> v;
    string field;
    istringstream sin(line);

    while (getline(sin, field, sepSymbol))
        v.push_back(field);

    return v;
}

double getElapsedTime(double start_tic)
{
    return 1000 * ((double) cv::getTickCount() - start_tic) / cv::getTickFrequency();
}

Matrix3d getSkewMatrix3(Vector3d w)
{
    Matrix3d r;
    r << 0, -w(2), w(1),
        w(2), 0, -w(0),
        -w(1), w(0), 0;
    return r;
}

VectorXd interpolateVectorXd(VectorXd v1, VectorXd v2, double step, double steps)
{
    return (VectorXd(v1 + (step / steps) * (v2 - v1)));
}

void project3DPoints(cv::Mat rMatrix, cv::Mat tVec, cv::Mat k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos)
{
    Mat rVec;                           // Rotation vector
    Rodrigues(rMatrix, rVec);

    Mat distCoeffs;

//    for (int i = 0; i < 3; i++)
//        for (int j = 0; j < 3; j++)
//            cout << "K[" << i << " ," << j << "]: " << k.at<double>(i, j)<< endl;

//    for (int j = 0; j < 3; j++)
//        cout << "rVec["<< j << "]: " << rVec.at<double>(0, j)<< endl;

    projectPoints(froms, rVec, tVec, k, distCoeffs, tos);
}

/*
 pose: 3x4 matrix, R33 | T31
 k: camera intrinsic parameters, 3X3 matrix
 */
void project3DPoints(cv::Mat pose, cv::Mat k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos)
{
    Mat tVec(3, 1, cv::DataType<double>::type);         // Translation vector
    Mat rMatrix(3, 3, cv::DataType<double>::type);
    for (int i = 0; i < 3; i++)
    {
        tVec.at<double>(i) = pose.at<double>(i, 3);
        for (int j = 0; j < 3; j++)
            rMatrix.at<double>(i, j) = pose.at<double>(i, j);
    }

    project3DPoints(rMatrix, tVec, k, froms, tos);
}

void project3DPoints(Eigen::Matrix3d r, Eigen::Vector3d t, Eigen::Matrix3d k, std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos)
{
    Mat tVec(3, 1, cv::DataType<double>::type);
    Mat rMatrix(3, 3, cv::DataType<double>::type);
    Mat kMatrix(3, 3, cv::DataType<double>::type);

    eigen2cv(k, kMatrix);
    eigen2cv(r, rMatrix);
    eigen2cv(t, tVec);

    project3DPoints(rMatrix, tVec, kMatrix, froms, tos);
}

/*
 eulers: 3x1 matrix, including roll, pitch, yaw in radius
 tVec: 3x1 matrix of translation vector
 k: camera intrinsic parameters, 3X3 matrix
 */
void project3DPoints(std::vector<cv::Point3f> froms, std::vector<cv::Point2f> &tos, cv::Mat eulers, cv::Mat tVec, cv::Mat k)
{
    Mat rMatrix(3, 3, cv::DataType<double>::type);
    eulers2Mat(eulers.at<double>(2), eulers.at<double>(1), eulers.at<double>(0), rMatrix);
    project3DPoints(rMatrix, tVec, k, froms, tos);
}

Eigen::Matrix3d eulers2Matrix(double yawZ, double pitchY, double rollX)
{
    Eigen::Matrix3d rRot;

    rRot(0, 0) = cos(yawZ) * cos(pitchY);
    rRot(1, 0) = sin(yawZ) * cos(pitchY);
    rRot(2, 0) = -sin(pitchY);

    rRot(0, 1) = cos(yawZ) * sin(pitchY) * sin(rollX) - sin(yawZ) * cos(rollX);
    rRot(1, 1) = sin(yawZ) * sin(pitchY) * sin(rollX) + cos(yawZ) * cos(rollX);
    rRot(2, 1) = cos(pitchY) * sin(rollX);

    rRot(0, 2) = cos(yawZ) * sin(pitchY) * cos(rollX) + sin(yawZ) * sin(rollX);
    rRot(1, 2) = sin(yawZ) * sin(pitchY) * cos(rollX) - cos(yawZ) * sin(rollX);
    rRot(2, 2) = cos(pitchY) * cos(rollX);

    return rRot;
}

void eulers2Mat(double yawZ, double pitchY, double rollX, cv::Mat &rRot)
{
#if 0
    rRot.at<double>(0, 0) = cos(yawZ) * cos(pitchY);
    rRot.at<double>(1, 0) = sin(yawZ) * cos(pitchY);
    rRot.at<double>(2, 0) = -sin(pitchY);

    rRot.at<double>(0, 1) = cos(yawZ) * sin(pitchY) * sin(rollX) - sin(yawZ) * cos(rollX);
    rRot.at<double>(1, 1) = sin(yawZ) * sin(pitchY) * sin(rollX) + cos(yawZ) * cos(rollX);
    rRot.at<double>(2, 1) = cos(pitchY) * sin(rollX);

    rRot.at<double>(0, 2) = cos(yawZ) * sin(pitchY) * cos(rollX) + sin(yawZ) * sin(rollX);
    rRot.at<double>(1, 2) = sin(yawZ) * sin(pitchY) * cos(rollX) - cos(yawZ) * sin(rollX);;
    rRot.at<double>(2, 2) = cos(pitchY) * cos(rollX);
#else
    Matrix<double, 3, 3> r = eulers2Matrix(yawZ, pitchY, rollX);
    eigen2cv(r, rRot);
#endif
}

void twoaxisrot(double r11, double r12, double r21, double r31, double r32, double res[])
{
    res[0] = atan2(r11, r12);
    res[1] = acos(r21);
    res[2] = atan2(r31, r32);
}

void threeaxisrot(double r11, double r12, double r21, double r31, double r32, double res[])
{
    res[0] = atan2(r31, r32);
    res[1] = asin(r21);
    res[2] = atan2(r11, r12);
}

Eigen::Vector3d quaternion2Eulers(Eigen::Quaterniond qt, RotSeq rotSeq) // euler (yaw, pitch, roll)
{
    Quaterniond qn = qt.normalized();
    uContainer q(qn.w(), qn.x(), qn.y(), qn.z());
    double res[3];

    switch (rotSeq)
    {
    case zyz:
        twoaxisrot(2 * (q.y * q.z - q.w * q.x),
                   2 * (q.x * q.z + q.w * q.y),
                   q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                   2 * (q.y * q.z + q.w * q.x),
                   -2 * (q.x * q.z - q.w * q.y),
                   res);
        break;

    case zxy:
        threeaxisrot(-2 * (q.x * q.y - q.w * q.z),
                     q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                     2 * (q.y * q.z + q.w * q.x),
                     -2 * (q.x * q.z - q.w * q.y),
                     q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                     res);
        break;

    case zxz:
        twoaxisrot(2 * (q.x * q.z + q.w * q.y),
                   -2 * (q.y * q.z - q.w * q.x),
                   q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                   2 * (q.x * q.z - q.w * q.y),
                   2 * (q.y * q.z + q.w * q.x),
                   res);
        break;

    case yxz:
        threeaxisrot(2 * (q.x * q.z + q.w * q.y),
                     q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                     -2 * (q.y * q.z - q.w * q.x),
                     2 * (q.x * q.y + q.w * q.z),
                     q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                     res);
        break;

    case yxy:
        twoaxisrot(2 * (q.x * q.y - q.w * q.z),
                   2 * (q.y * q.z + q.w * q.x),
                   q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                   2 * (q.x * q.y + q.w * q.z),
                   -2 * (q.y * q.z - q.w * q.x),
                   res);
        break;

    case yzx:
        threeaxisrot(-2 * (q.x * q.z - q.w * q.y),
                     q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                     2 * (q.x * q.y + q.w * q.z),
                     -2 * (q.y * q.z - q.w * q.x),
                     q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                     res);
        break;

    case yzy:
        twoaxisrot(2 * (q.y * q.z + q.w * q.x),
                   -2 * (q.x * q.y - q.w * q.z),
                   q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                   2 * (q.y * q.z - q.w * q.x),
                   2 * (q.x * q.y + q.w * q.z),
                   res);
        break;

    case xyz:
        threeaxisrot(-2 * (q.y * q.z - q.w * q.x),
                     q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                     2 * (q.x * q.z + q.w * q.y),
                     -2 * (q.x * q.y - q.w * q.z),
                     q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                     res);
        break;

    case xyx:
        twoaxisrot(2 * (q.x * q.y + q.w * q.z),
                   -2 * (q.x * q.z - q.w * q.y),
                   q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                   2 * (q.x * q.y - q.w * q.z),
                   2 * (q.x * q.z + q.w * q.y),
                   res);
        break;

    case xzy:
        threeaxisrot(2 * (q.y * q.z + q.w * q.x),
                     q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z,
                     -2 * (q.x * q.y - q.w * q.z),
                     2 * (q.x * q.z + q.w * q.y),
                     q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                     res);
        break;

    case xzx:
        twoaxisrot(2 * (q.x * q.z - q.w * q.y),
                   2 * (q.x * q.y + q.w * q.z),
                   q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                   2 * (q.x * q.z + q.w * q.y),
                   -2 * (q.x * q.y - q.w * q.z),
                   res);
        break;

    case zyx:
    default:
        threeaxisrot(2 * (q.x * q.y + q.w * q.z),
                     q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z,
                     -2 * (q.x * q.z - q.w * q.y),
                     2 * (q.y * q.z + q.w * q.x),
                     q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
                     res);
        break;
    }

    return (Vector3d(res[0], res[1], res[2]));
}

Quaterniond eulers2Quaterniond(Vector3d e)
{
    return eulers2Quaterniond(e(2), e(1), e(0));
}

Quaterniond eulers2Quaterniond(double yaw, double pitch, double roll)
{
#if 0
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaterniond q;
    q.w() = cr * cp * cy + sr * sp * sy;
    q.x() = sr * cp * cy - cr * sp * sy;
    q.y() = cr * sp * cy + sr * cp * sy;
    q.z() = cr * cp * sy - sr * sp * cy;
#else
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
#endif

    return q.normalized();
}

Matrix3d quaternion2Matrix(Quaterniond q)
{
    Quaterniond qn = q.normalized();
    double w = qn.w(), x = qn.x(), y = qn.y(), z = qn.z();

    double Rxx = 1 - 2 * (y * y + z * z);
    double Rxy = 2 * (x * y - z * w);
    double Rxz = 2 * (x * z + y * w);

    double Ryx = 2 * (x * y + z * w);
    double Ryy = 1 - 2 * (x * x + z * z);
    double Ryz = 2 * (y * z - x * w);

    double Rzx = 2 * (x * z - y * w);
    double Rzy = 2 * (y * z + x * w);
    double Rzz = 1 - 2 * (x * x + y * y);

    Matrix3d r;
    r << Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz;

    return r;
}

int getRandomUnifrom(int lower, int upper)
{
    std::random_device rd;                                  // obtain a random number from hardware
    std::mt19937 gen(rd());                                 // seed the generator
    std::uniform_int_distribution<> distr(lower, upper);    // define the range
    return distr(gen);
}

double getRandomNormal(double mean, double sigma, double maximum)
{
    std::random_device rd;                                  // obtain a random number from hardware
    std::mt19937 gen(rd());                                 // seed the generator
    std::normal_distribution<double> distr(mean, sigma);
    double val = distr(gen);

    if (maximum > mean)
    {
        if (val > maximum)
            return maximum;

        if (val < -maximum)
            return -maximum;
    }

    return val;
}

std::string itosWith0(long number, int numberOf0)
{
    std::stringstream ss;
    ss << std::setw(numberOf0) << std::setfill('0') << number;
    return ss.str();
}
