//
//  SettingConstant.h
//  FrontendTracking
//
//  Created by Lau Yo-Chung on 2022/5/29.
//  Copyright © 2022 Lau Yo-Chung. All rights reserved.
//

#ifndef SettingConstant_h
#define SettingConstant_h

#include <iostream>

struct SharingData
{
    //cv::Scalar _poseColorSF = cv::Scalar(0, 255, 255);      // yellow
    cv::Scalar _poseColorSF = cv::Scalar(255, 255, 255);    // white
    cv::Scalar _poseColorGT = cv::Scalar(255, 0, 0);        // blue

    bool _isBSCM = true;
    bool _isBSCMOnlyV = !_isBSCM;
    bool _isGTPoseBias = false;

    std::string _baseSimPath = "/Users/sherlock/Downloads/simulation/results/";
    std::string _baseFramePath = _baseSimPath + "frames/";
    std::string _baseBoxPath = _baseFramePath + "boxes/";

    std::string _dfcmPath = _baseSimPath + "Requirement.txt";

    /* imu.txt / imu_noise.txt
     time_in_second time_in_ns ax ay az wx wy wz
     ex: 13 378000000 -0.046288 -8.505662 -4.874295 -0.005761 -0.004691 -0.007285
    */
    std::string _imuPath = _baseSimPath + "imu.txt";
    std::string _imu2Path = _baseSimPath + "imu2.txt";


    /*
     time_in_second time_in_ns GT_pose_T_(2 ~ 4) GT_pose_E_(5 ~ 7)
     ex: 1 226000000 0.000660 0.000837 1.398666 -2.093974 0.000592 0.000281
    */
    std::string _posePath = _baseSimPath + "pose.txt";

    /*
     frameName time_in_second time_in_ns GT_pose_T_(3 ~ 5) GT_pose_E_(6 ~ 8)f
     ex: 00000033 1 226000000 0.000660 0.000837 1.398666 -2.093974 0.000592 0.000281
    */
    std::string _imagePosePath = _baseSimPath + "image_pose.txt";
    std::string _imagePose2Path = _baseSimPath + "image_pose2.txt";

    /*
     frameName tracking_pose_T_(1 ~ 3) tracking_pose_E_(4 ~ 6) GT_pose_T_(7 ~ 9) GT_pose_E_(10 ~ 12) isTracking(13)
     ex: 00000070.jpg 0.026284 0.033375 1.346805 -2.078244 0.022316 0.010827 0.022065 0.027973 1.355389 -2.080141 0.019722 0.009545 1
    */
    std::string _trackingPosePath = _baseSimPath + "tracking_pose.txt";

    std::string _pcmAnzPath = _baseSimPath + "pcm_analysis.txt";

    /*
     frameName error_T_(1 ~ 3) error_E_(4 ~ 6) error_average_projection(in pixel) isTracking
     ex: 00000027.jpg 0.000014 0.000017 1.399971 -2.094388 0.000013 0.000006 0.000010 0.000012 1.399980 -2.094391 0.000009 0.000004 1
    */
    std::string _trackingErrorPath = _baseSimPath + "tracking_error.txt";

    /*
     last_frame_index current_frame_index error_average_projection(in pixel)
     ex: 1563 1564 1.411221
    */
    std::string _ajPoseDiffPath = _baseSimPath + "adjacent_pose_difference.txt";

    std::string _modelCornerPath = _baseSimPath + "corner.txt";
    std::string _modelConfigPath = _baseSimPath + "model_config.txt";

    std::string _intrinsicPath = _baseSimPath + "intrinsic.txt";
    std::string _biasPath = _baseSimPath + "bias.txt";

    double _startTime = 0;
    double _duration = 30;
    double _endTime = _startTime + _duration;

    int _poseRate = 1000;
    int _imuRate = 200;
    int _frameRate = 30;
    // relationships: _frameRate <= _imuRate <= _poseRate

    double _timeRatio = 2.46;

    double _timeFront = 5.0 / 1000.0;

    /*
     frame size: 640 x 480 x 3 (RGB) x 1 B / 10 (JPEG compression ratio) / 1024 = 90 KB, so we take take it as 100 KB including some metadata like pose
     5G bandwidth: 50 Mbps
     data transmition time: (50 * 8) / (100 * 1024) * 1000 = 15.625 ms
     5G propagation delay: 10 ms (x 2)
     extra delay: 30 ms
     so, total delay time: 10 x 2 + 15.625 + 30 = 65.625 ms
    */
    double _extraDelay = 30.0;                                                              // in ms
    double _baseTimeBack = 35.625 / 1e3;                                                    // in second
    //double _baseTimeBack = 10 / 1e3;                                                    // in second
    double _timeBack = _baseTimeBack + _extraDelay / 1e3;                                   // in second
    //double _timeBack = 250 / 1000.0;                                                      // in second
    bool _isFixedExtraDelay = false;

    double _biasCorrectStartTime = 0.0;                                                                // in second

    void _resetPathes(std::string basePath)
    {
        _baseSimPath = basePath;
        _baseFramePath = _baseSimPath + "frames/";
        _baseBoxPath = _baseFramePath + "boxes/";
        _dfcmPath = _baseSimPath + "Requirement.txt";
        _imuPath = _baseSimPath + "imu.txt";
        _imu2Path = _baseSimPath + "imu2.txt";
        _posePath = _baseSimPath + "pose.txt";
        _imagePosePath = _baseSimPath + "image_pose.txt";
        _imagePose2Path = _baseSimPath + "image_pose2.txt";
        _trackingPosePath = _baseSimPath + "tracking_pose.txt";
        _trackingErrorPath = _baseSimPath + "tracking_error.txt";
        _ajPoseDiffPath = _baseSimPath + "adjacent_pose_difference.txt";
        _pcmAnzPath = _baseSimPath + "pcm_analysis.txt";
        _modelCornerPath = _baseSimPath + "corner.txt";
        _modelConfigPath = _baseSimPath + "model_config.txt";
        _intrinsicPath = _baseSimPath + "intrinsic.txt";
        _biasPath = _baseSimPath + "bias.txt";
    }
};

#endif /* SettingConstant_h */
