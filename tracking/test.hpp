//
//  test.hpp
//  FrontendTracking
//
//  Created by Lau Yo-Chung on 2021/3/21.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#ifndef test_hpp
#define test_hpp

#include <iostream>

#include "DataStructure.h"
#include "Setting.h"

void testEigen();
void testFile(std::string fileName);
void testFrames(std::string fileName, std::string basePath);
void testIMU(SharingData sharingData, bool isByRK4);
void testPCM(cv::Mat &rFrame, cv::Mat &frame);
void testGeometry();
void testGZProjection();
void testBiasW();
void testDuck();

#endif /* test_hpp */
