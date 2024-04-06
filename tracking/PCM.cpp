//
//  PCMCore.cpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/2/20.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

#include "Utility.hpp"
#include "PCM.hpp"

bool _isNativePrj = false;
double _AREA_UNDER_TRACKING = _AREA_UNDER_TRACKING_DEFAULT;
double _Pxe = _Pxe_DEFAULT;
double _Pxm = _Pxm_DEFAULT;
double _OFFSET_LEGAL_POSE = 17.0;
PCMRecord _record;

using namespace std;
using namespace cv;
using namespace Eigen;

std::vector<cv::Point2f> getBoundingbox2DPoints(ObjectInfo object, Eigen::Matrix3d cameraK, BasicPose pose)
{
    vector<Point3f> boxes;
    vector<Point2f> boxesP;

    if (_isNativePrj)
    {
        for (int i = 0; i < object.boxesRT.size(); i++)
        {
            Vector3d p = object.boxesRT[i];
            boxes.push_back(Point3f(p(0), p(1), p(2)));
        }

        project3DPoints(pose.r, pose.t, cameraK, boxes, boxesP);
    }
    else
    {
        for (int i = 0; i < object.boxesRT.size(); i++)
        {
            Vector3d p = object.boxesRT[i] - pose.t;
            boxes.push_back(Point3f(p(0), p(1), p(2)));
        }

        project3DPoints(pose.r.inverse(), Vector3d(), cameraK, boxes, boxesP);
    }

    return boxesP;
}

double getBoundingbox2DPointsDiff(ObjectInfo *object, Eigen::Matrix3d cameraK, BasicPose pose1, BasicPose pose2)
{
    vector<Point2f> boxesNow = getBoundingbox2DPoints(*object, cameraK, pose1);
    vector<Point2f> boxesLast = getBoundingbox2DPoints(*object, cameraK, pose2);

    double dis = 0;
    for (int i = 0; i < boxesNow.size(); i++)
        dis += norm(boxesNow[i] - boxesLast[i]);
    dis /= boxesNow.size();

    return dis;
}

bool isPoseAcceptable(ObjectInfo object, Eigen::Matrix3d cameraK, BasicPose currentPose, BasicPose lastPose, string fileName = "")
{
    double diff = getBoundingbox2DPointsDiff(&object, cameraK, currentPose, lastPose);
    if (diff > _record.largestOffset)
    {
        _record.largestOffset = diff;
        _record.largestOffsetName = fileName;
    }

    _record.sumOffset += diff;
    _record.processed++;

//    cout << "The offset between two frames: " << diff << " / " << _OFFSET_LEGAL_POSE << endl;
//    cout << "The average offset between two frames: " << _record.average() << endl;
//    cout << "The largest offset between two frames: " << _record.largestOffset << endl;
    
    return diff < _OFFSET_LEGAL_POSE;
}

bool isUnderTracking(ObjectInfo object, Matrix3d cameraK, FrameInfo frame, IMUPose pose)
{
    vector<Point2f> boxesP = getBoundingbox2DPoints(object, cameraK, BasicPose(pose.r, pose.p));

//    cout << "PCM of frame: " << frame.index << endl;
    Point2f leftBottom(INFINITY, INFINITY), rightUp(0, 0);
    for (int i = 0; i < boxesP.size(); i++)
    {
        rightUp.x = MAX(rightUp.x, boxesP[i].x);
        rightUp.y = MAX(rightUp.y, boxesP[i].y);

        leftBottom.x = MIN(leftBottom.x, boxesP[i].x);
        leftBottom.y = MIN(leftBottom.y, boxesP[i].y);

//        cout << "3D point: (" << object.boxesRT[i](0) << "," << object.boxesRT[i](1) << "," << object.boxesRT[i](2) << ")" << endl;
//        cout << "2D point: (" << boxesP[i].x << "," << boxesP[i].y << ")" << endl;
    }

    rightUp.x = MIN(MAX(rightUp.x, 0), frame.frame.cols - 1);
    rightUp.y = MIN(MAX(rightUp.y, 0), frame.frame.rows - 1);

    leftBottom.x = MIN(MAX(leftBottom.x, 0), frame.frame.cols - 1);
    leftBottom.y = MIN(MAX(leftBottom.y, 0), frame.frame.rows - 1);

    vector<Point2f> rec;
    rec.push_back(cv::Point2f(leftBottom.x, rightUp.y));    // left up
    rec.push_back(cv::Point2f(rightUp.x, rightUp.y));       // right up
    rec.push_back(cv::Point2f(rightUp.x, leftBottom.y));    // right bottom
    rec.push_back(cv::Point2f(leftBottom.x, leftBottom.y)); // left bottom
    double area = contourArea(rec);

//    for(int i = 0; i < rec.size(); i++)
//        cout << "Rectangle points: (" << rec[i].x << "," << rec[i].y << ")" << endl;
//
//    cout << "area of Rectangle of 2D points: " << area << endl;

    return area >= _AREA_UNDER_TRACKING;
}

bool doPCM(ObjectInfo object, Eigen::Matrix3d cameraK, FrameInfo frame, IMUPose pose, BasicPose lastPose)
{
//    _OFFSET_LEGAL_POSE = 17;
//    cout << "_OFFSET_LEGAL_POSE: " << _OFFSET_LEGAL_POSE << endl;
    
    if (isUnderTracking(object, cameraK, frame, pose))
    {
        double diff = getBoundingbox2DPointsDiff(&object, cameraK, BasicPose(pose.r, pose.p), BasicPose(frame.r, frame.p));
       // cout << "The offset between GT: " << diff << endl;
        return lastPose.isActive? isPoseAcceptable(object, cameraK, BasicPose(pose.r, pose.p), lastPose, frame.fileName) : true;
    }

    return false;;
}

void setProjectionType (bool isNativePrj)
{
    _isNativePrj = isNativePrj;
}

void setTHR2D(double pxe, double pxm)
{
    _Pxe = pxe;
    _Pxm = pxm;

    cout << "set (Pxe, Pxm) as " << "(" << _Pxe << ", " << _Pxm << ")" << endl;
}

void setTHRArea(double area)
{
    _AREA_UNDER_TRACKING = area;
}

void resetPCM()
{
    setTHRArea(_AREA_UNDER_TRACKING_DEFAULT);
    setTHR2D(_Pxe_DEFAULT, _Pxm_DEFAULT);
    _record.reset();
}

PCMRecord getPCMRecord()
{
    return _record;
}
