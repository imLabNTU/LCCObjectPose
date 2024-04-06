//
//  IOTool.hpp
//  FrontendTracking
//
//  Created by Lau Yo-Chung on 2022/5/28.
//  Copyright © 2022 Lau Yo-Chung. All rights reserved.
//

#ifndef IOTool_hpp
#define IOTool_hpp

#include <iostream>

#include "SettingConstant.h"
#include "DataStructure.h"
#include "IMUPose.hpp"

void loadObjectInfo (ObjectInfo *obj, std::string cornerPath, std::string configPath, bool *isNativeProj);
void loadCameraInfo (Eigen::Matrix3d *K, std::string intrinsicPath);

bool readFrameInfo(std::ifstream &fp, FrameInfo &fi, SharingData sharingData, bool isFrameNeeded , bool is2ShowLog = false);
bool readIMUInfo(std::ifstream &fp, IMUPose &imu, bool is2ShowLog = false);

void loadDefaultBiasFromFile(IMUPose &imu, std:: string fileName);
void loadDefaultBiasFromData(IMUPose &imu, Eigen::Vector3d a, Eigen::Vector3d w);

void drawBoundingbox(std::string srcFramePath, std::string dstFramePath, BasicPose pose, ObjectInfo object, Eigen::Matrix3d cameraK, cv::Scalar color = cv::Scalar(255, 255, 255));

void addDurationAndNextTime(std::string srcPath, std::string dstPath, int timeIndex);

void noiseFramePose(FrameInfo &fi, bool isAlways = false);

#endif /* IOTool_hpp */
