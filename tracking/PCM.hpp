//
//  PCMCore.hpp
//  PCM
//
//  Created by Lau Yo-Chung on 2021/2/20.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#ifndef PCMCore_hpp
#define PCMCore_hpp

#include <iostream>

#include "DataStructure.h"
#include "IMUPose.hpp"

#define _AREA_UNDER_TRACKING_DEFAULT        (64.0 * 48.0)
#define _Pxe_DEFAULT                        (5.0)
#define _Pxm_DEFAULT                        (5.0)

std::vector<cv::Point2f> getBoundingbox2DPoints(ObjectInfo object, Eigen::Matrix3d cameraK, BasicPose pose);

double getBoundingbox2DPointsDiff(ObjectInfo *object, Eigen::Matrix3d cameraK, BasicPose pose1, BasicPose pose2);

bool doPCM(ObjectInfo object, Eigen::Matrix3d cameraK, FrameInfo frame, IMUPose pose, BasicPose lastPose);

void setProjectionType (bool isNativePrj);

void setTHR2D(double pxe, double pxm);
void setTHRArea(double area);

void resetPCM();

struct PCMRecord
{
    PCMRecord()
    {
    };

    void reset()
    {
        largestOffsetName = "";
        largestOffset = -1;
        sumOffset = 0;
        processed = 0;
    };

    double average()
    {
        return processed > 0? sumOffset / processed : 0;;
    }

    std::string largestOffsetName = "";
    double largestOffset = -1;
    double sumOffset = 0;
    long processed = 0;
};

PCMRecord getPCMRecord();

#endif /* PCMCore_hpp */
