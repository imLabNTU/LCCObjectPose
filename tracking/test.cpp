//
//  test.cpp
//  FrontendTracking
//
//  Created by Lau Yo-Chung on 2021/3/21.
//  Copyright © 2021 Lau Yo-Chung. All rights reserved.
//

#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "Utility.hpp"
#include "IOTool.hpp"
#include "IMUPose.hpp"
#include "PCM.hpp"
#include "test.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

void testEigen()
{
    Matrix2d a;
    a << 1, 2,
        3, 4;
    a(0, 0) = -1;
    MatrixXd b(2, 2);
    b << 2, 3,
        1, 4;
    cout << "a + b =\n"
         << a + b << endl;
    cout << "a - b =\n"
         << a - b << endl;
    cout << "Doing a += b;" << endl;
    a += b;
    cout << "Now a =\n"
         << a << endl;
    cout << "a^T=  " << a.transpose() << endl;
    cout << "a*b= " << a * b << endl;
    Vector3d v(1, 2, 3);
    Vector3d w(1, 0, 0);
    //   cout << "v * 2 + w - v =\n" << v * 2 + w - v << endl;
    Vector3d vip = interpolateVectorXd(v, w, 3, 10);
    cout << "vip" << vip << endl;
}

void testFile(std::string fileName)
{
    ifstream inFile;
    string line, field;

    inFile.open(fileName);
    getline(inFile, line);

    vector<string> data(getLineSub(line, ' '));
}

void testFrames(std::string fileName, std::string basePath)
{
    if (fileName.length() > 0 )
    {
        ifstream fp;
        fp.open(fileName);

        while (1)
        {
            FrameInfo fi;
            SharingData sharingData; sharingData._baseFramePath = basePath; sharingData._baseBoxPath = "";
            if (!readFrameInfo(fp, fi, sharingData, true, true))
                break;
        };

        fp.close();
    }
}

#define _ResetPose
#define _ResetTime (65.0 * 1.0 / 1e3)   // (1.0 * 2 / 60.0 )
void testIMU(SharingData sharingData, bool isByRK4)
{
    BasicPose _lastPose;
    double errs = 0;
    long count = 0;
    bool is1stCorrect = true, is1stFrame = true;

    Vector3d averageV = Vector3d(0, 0, 0);
    double time4averageV = 0;
    Matrix3d _cameraK;
    ObjectInfo _object;

    bool isNativePrj = false;
    loadCameraInfo(&_cameraK, sharingData._intrinsicPath);
    loadObjectInfo(&_object, sharingData._modelCornerPath, sharingData._modelConfigPath, &isNativePrj);
    setProjectionType(isNativePrj);

    ifstream imuMetaFile, imagePoseMetaFile;
    imuMetaFile.open(sharingData._imu2Path);
    imagePoseMetaFile.open(sharingData._imagePose2Path);

    FrameInfo frameInfo; frameInfo.rate = sharingData._frameRate; frameInfo.period = 1e3 / (double) frameInfo.rate;
    while (1)
    {
        readFrameInfo(imagePoseMetaFile, frameInfo, sharingData, true);

        if (!frameInfo.status || frameInfo.time >= sharingData._endTime)
        {
            imuMetaFile.close();
            imagePoseMetaFile.close();
            return;
        }

        if (frameInfo.time >= sharingData._startTime)
            break;
    }
    double startTimeReal = frameInfo.time;
    FrameInfo lastFrame = frameInfo;
    _lastPose = BasicPose(eulers2Matrix(lastFrame.e(2), lastFrame.e(1), lastFrame.e(0)), lastFrame.p);

    IMUPose impPose; impPose.rate = sharingData._imuRate; impPose.period = 1e3 / (double) impPose.rate;
    impPose.p = frameInfo.p;
    impPose.r = eulers2Matrix(frameInfo.e(2), frameInfo.e(1), frameInfo.e(0));
    Quaterniond qn = Quaterniond (impPose.r);
    impPose.q(0) = qn.w(); impPose.q(1) = qn.x(); impPose.q(2) = qn.y(); impPose.q(3) = qn.z();
    loadDefaultBiasFromFile(impPose, sharingData._biasPath);

    system(("rm -r -f " + sharingData._baseBoxPath).c_str());
    system(("mkdir " + sharingData._baseBoxPath).c_str());

    double timeReset = frameInfo.time;
    double lastTime = frameInfo.time;
    while(1)
    {
        cout << "(impPose.v.norm, frameInfo.index): (" <<  impPose.v.norm() * 100 << ", " << frameInfo.index << ")" << endl;
        while (1)
        {
            readIMUInfo(imuMetaFile, impPose);

            if (!impPose.status || impPose.time > sharingData._endTime)
            {
                imuMetaFile.close();
                imagePoseMetaFile.close();
                break;
            }

            if (impPose.time < startTimeReal)
                continue;

            Vector3d e = quaternion2Eulers(Quaterniond(impPose.r));
            Vector3d p = impPose.p;

//            cout << "duration: " << fixed << setprecision(6) << impPose.duration / 1e3 << "s" << endl;
//            cout << "a: " << impPose.a(0) << ", " << impPose.a(1) << ", " << impPose.a(2) << ")" << endl;
//            cout << "w: " << impPose.w(0) << ", " << impPose.w(1) << ", " << impPose.w(2) << ")" << endl;
//
//            cout << "Before imu updating..." << endl;
//            cout << "IMU time: " << impPose.time << fixed << setprecision(6) << endl;
//            cout << "IMU translation: " << "(" << p(0) << ", " << p(1) << ", " << p(2) << ")" << endl;
//            cout << "IMU eulers: " << "(" << e(0) << ", " << e(1) << ", " << e(2) << ")" << endl;

            cout << "impPose.time: " << impPose.time << endl;

            if (isByRK4)
                impPose.rk4(impPose.w, impPose.a, impPose.duration / 1e3);
            else
                impPose.update(impPose.w, impPose.a, impPose.duration / 1e3);

            e = quaternion2Eulers(Quaterniond(impPose.r));
            p = impPose.p;

//            cout << "After imu updating..." << endl;
//            cout << "IMU time: " << impPose.time << fixed << setprecision(6) << endl;
//            cout << "IMU translation: " << "(" << p(0) << ", " << p(1) << ", " << p(2) << ")" << endl;
//            cout << "IMU eulers: " << "(" << e(0) << ", " << e(1) << ", " << e(2) << ")" << endl;

            if (impPose.time + impPose.toNextTime / 1e3 > frameInfo.time)
                break;
        }

        // compare pose
        Vector3d e = quaternion2Eulers(Quaterniond(impPose.r));
        Vector3d p = impPose.p;

//        cout << "GT pose time: " << frameInfo.time << fixed << setprecision(6) << endl;
//        cout << "IMU time: " << impPose.time << endl;
//        cout << "GT eulers: " << "(" << frameInfo.e(0) << ", " << frameInfo.e(1) << ", " << frameInfo.e(2) << ")" << endl;
//        cout << "IMU eulers: " << "(" << e(0) << ", " << e(1) << ", " << e(2) << ")" << endl;
//        cout << "GT translation: " << "(" << frameInfo.p(0) << ", " << frameInfo.p(1) << ", " << frameInfo.p(2) << ")" << endl;
//        cout << "IMU translation: " << "(" << p(0) << ", " << p(1) << ", " << p(2) << ")" << endl;
//        cout << " IMU translation diff:" << endl << fixed << setprecision(6) << (impPose.p - frameInfo.p) << endl;

        BasicPose pose = BasicPose(eulers2Matrix(e(2), e(1), e(0)), p);

        string srcPath = sharingData._baseFramePath + frameInfo.fileName;
        if (srcPath.compare(srcPath.length() - 4, srcPath.length(), ".jpg"))
            srcPath += ".jpg";

        string dstPath = sharingData._baseBoxPath + frameInfo.fileName;
        if (dstPath.compare(dstPath.length() - 4, dstPath.length(), ".jpg"))
            dstPath += ".jpg";

        drawBoundingbox(srcPath, dstPath, pose, _object, _cameraK, sharingData._poseColorSF);
        Vector3d ge = frameInfo.p;
        BasicPose poseGT = BasicPose(eulers2Matrix(frameInfo.e(2), frameInfo.e(1), frameInfo.e(0)), ge);
        drawBoundingbox(dstPath, dstPath, poseGT, _object, _cameraK, sharingData._poseColorGT);

//        cout << "dt: " << frameInfo.time - impPose.time << endl;
        double dis = getBoundingbox2DPointsDiff(&_object, _cameraK, pose, poseGT);
        cout << frameInfo.time << "/" << frameInfo.index << ": bounding box error: " << dis << endl;
        cout << "(imu v.norm, imu time): (" <<  impPose.v.norm() * 100 << ", " << impPose.time << ")" << endl;
        errs += dis; count++;
        _lastPose = poseGT;

//       if (dis > 1)
//       {
//           cout << "averageV:" << averageV << endl;
//           cout << "imu v: " << impPose.v << endl;
//       }

        bool poseUpdated = false;
    #ifdef _ResetPose
        #define _local_diff_thrd                (0.0)
        #define _local_bias_divider_a           (2)
        #define _local_bias_divider_w           (1)

        if ( frameInfo.time - timeReset >= _ResetTime)
        {
            cout << "Backend pose estimation!" << endl;

            double dt =  frameInfo.time - lastTime;
            Vector3d dp = frameInfo.p - impPose.p;
            time4averageV = dt;
            Vector3d dv = dp / dt;
            Vector3d a = dv / dt;

            Matrix3d dr = impPose.r.inverse() * frameInfo.r;
            Quaterniond dq = Quaterniond(dr);
            Vector3d de = quaternion2Eulers(dq);
            Vector3d w = de / dt;

            if (is1stCorrect) {
                cout << "Reset pose!" << endl;
                poseUpdated = true;

                impPose.r = eulers2Matrix(frameInfo.e(2), frameInfo.e(1), frameInfo.e(0));
                impPose.p = frameInfo.p;

                impPose._v = impPose.v = dv;
//                impPose.bias_a = impPose.bias_w =  Vector3d(0, 0, 0);
                impPose.bias_a = impPose.bias_a + a / _local_bias_divider_a;
                impPose.bias_w = impPose.bias_w + w / _local_bias_divider_w;

                is1stCorrect = false;
                lastTime = frameInfo.time;
            } else {
                if (dis >= _local_diff_thrd)
                {
                    cout << "Reset pose!" << endl;
                    poseUpdated = true;

                    impPose.r = eulers2Matrix(frameInfo.e(2), frameInfo.e(1), frameInfo.e(0));
                    impPose.p = frameInfo.p;

                    impPose._v = impPose._v + dv;
                    impPose.v = impPose.v + dv;

                    impPose.bias_a = impPose.bias_a + a / _local_bias_divider_a;
                    impPose.bias_w = impPose.bias_w + w / _local_bias_divider_w;

                    lastTime = frameInfo.time;
                }
            }

            timeReset = frameInfo.time;

//            if (frameInfo.index <= 82)
//                impPose._v = impPose.v = Vector3d(0, 0, 0);
//            cout << "new imu v: " << impPose.v << endl;

            Vector3d v = (frameInfo.p - lastFrame.p) / ((frameInfo.time - lastFrame.time));
//            cout << "delta frame V: " << v - averageV << endl;
            averageV = v;// is1stFrame? averageV : (averageV + v) / 2;
            lastFrame = frameInfo;
        }
    #endif

//        lastFrame = frameInfo;

        readFrameInfo(imagePoseMetaFile, frameInfo, sharingData, true);
        if (!frameInfo.status || frameInfo.time >= sharingData._endTime)
            break;

        if (0 && poseUpdated)
        {
            Vector3d dv = averageV - impPose.v;
            Vector3d a = dv / time4averageV;

            cout << "reset frame V: " << averageV.norm() * 100 << endl;

            impPose._v = impPose._v + dv;
            impPose.v = impPose.v + dv;

            impPose.bias_a = impPose.bias_a + a / _local_bias_divider_a;
        }

        is1stFrame = false;

//                    if (frameInfo.index <= 150)
//                    {
//                        impPose._v = impPose.v = Vector3d(0, 0, 0);
//                        impPose.bias_a = Vector3d(9.76183, 1.03468, 10.8867);
//                    }

        if (sharingData._isGTPoseBias)
        {
            BasicPose lastPose = BasicPose(frameInfo.r, frameInfo.p);
            noiseFramePose(frameInfo);

//            double dis = getBoundingbox2DPointsDiff(_object, _cameraK,  BasicPose(_nowFrame.r, _nowFrame.p), lastPose);
//            cout << "bounding box error: " << dis << endl;
        }
    }

    cout << "average error: " << errs / count << endl;


    imuMetaFile.close();
    imagePoseMetaFile.close();
}

void testPCM(cv::Mat &rFrame, cv::Mat &frame)
{
    cv::Mat image = rFrame;
    cv::Mat image2 = frame;

    vector<Point2f> tos;
    vector<Point3f> froms;
    froms.push_back(cv::Point3f(-0.033737, -0.061801, -0.045446));
    froms.push_back(cv::Point3f(-0.033737, -0.061801, 0.072011));
    froms.push_back(cv::Point3f(-0.033737, 0.065832, -0.045446));
    froms.push_back(cv::Point3f(-0.033737, 0.065832, 0.072011));
    froms.push_back(cv::Point3f(0.033274, -0.061801, -0.045446));
    froms.push_back(cv::Point3f(0.033274, -0.061801, 0.072011));
    froms.push_back(cv::Point3f(0.033274, 0.065832, -0.045446));
    froms.push_back(cv::Point3f(0.033274, 0.065832, 0.072011));

    vector<Point2f> gt;
    gt.push_back(cv::Point2f(332.82528011252606, 173.49186722365627));
    gt.push_back(cv::Point2f(323.3667158966044, 125.40668535608732));
    gt.push_back(cv::Point2f(397.3930556606747, 161.63712786464984));
    gt.push_back(cv::Point2f(393.3849622238311, 112.21991031697068));
    gt.push_back(cv::Point2f(336.5279893659714, 197.15268313965993));
    gt.push_back(cv::Point2f(326.97169082502995, 148.99942506328279));
    gt.push_back(cv::Point2f(403.8074402964834, 184.95541873584958));
    gt.push_back(cv::Point2f(400.1901449692861, 135.37867319566013));

    Mat k(3, 3, cv::DataType<double>::type);
    k.at<double>(0, 0) = 572.4114;
    k.at<double>(0, 1) = 0;
    k.at<double>(0, 2) = 325.2611;
    k.at<double>(1, 0) = 0;
    k.at<double>(1, 1) = 573.57043;
    k.at<double>(1, 2) = 242.04899;
    k.at<double>(2, 0) = 0;
    k.at<double>(2, 1) = 0;
    k.at<double>(2, 2) = 1;

    Mat pose(3, 4, cv::DataType<double>::type);
    pose.at<double>(0, 0) = 0.09506610035896301;
    pose.at<double>(0, 1) = 0.9833089709281921;
    pose.at<double>(0, 2) = -0.15512900054454803;
    pose.at<double>(0, 3) = 0.07173020392656326;
    pose.at<double>(1, 0) = 0.741595983505249;
    pose.at<double>(1, 1) = -0.17391300201416016;
    pose.at<double>(1, 2) = -0.647911012172699;
    pose.at<double>(1, 3) = -0.1490722894668579;
    pose.at<double>(2, 0) = -0.6640759706497192;
    pose.at<double>(2, 1) = -0.05344890058040619;
    pose.at<double>(2, 2) = -0.7457519769668579;
    pose.at<double>(2, 3) = 1.0606387853622437;

    double tic = (double)getTickCount();
    project3DPoints(pose, k, froms, tos);
    cout << endl
         << "project points in " << 1000 * ((double)getTickCount() - tic) / getTickFrequency() << " ms" << endl;

    Point2f leftBottom(INFINITY, INFINITY), rightUp(0, 0);
    unsigned int avError = 0;
    for (int i = 0; i < froms.size(); i++)
    {
        double error = cv::norm(gt[i] - tos[i]);
        cout << "3D point: (" << froms[i].x << "," << froms[i].y << "," << froms[i].z << ")" << endl;
        cout << "2D point: (" << tos[i].x << "," << tos[i].y << ")" << endl;
        cout << "2D point (GT): (" << gt[i].x << "," << gt[i].y << ")" << endl;
        cout << "project error: " << error << endl;
        avError += error;

        rightUp.x = MAX(rightUp.x, tos[i].x);
        rightUp.y = MAX(rightUp.y, tos[i].y);
        leftBottom.x = MIN(leftBottom.x, tos[i].x);
        leftBottom.y = MIN(leftBottom.y, tos[i].y);
    }
    avError = avError / froms.size();
    cout << "average project error: " << avError << endl;

    rightUp.x = MIN(MAX(rightUp.x, 0), image.cols - 1);
    rightUp.y = MIN(MAX(rightUp.y, 0), image.rows - 1);
    leftBottom.x = MIN(MAX(leftBottom.x, 0), image.cols - 1);
    leftBottom.y = MIN(MAX(leftBottom.y, 0), image.rows - 1);

    vector<Point2f> rec;
    rec.push_back(cv::Point2f(leftBottom.x, rightUp.y));    // left up
    rec.push_back(cv::Point2f(rightUp.x, rightUp.y));       // right up
    rec.push_back(cv::Point2f(rightUp.x, leftBottom.y));    // right bottom
    rec.push_back(cv::Point2f(leftBottom.x, leftBottom.y)); // left bottom
                                                            //    for(int i = 0; i < rec.size(); i++)
                                                            //        cout << "Rectangle points: (" << rec[i].x << "," << rec[i].y << ")" << endl;
    cout << "area of Rectangle of 2D points: " << cv::contourArea(rec) << endl;
    // area threshold? sherlock

    tic = (double)getTickCount();
    unsigned int area = cv::contourArea(tos);
    cout << endl
         << "calculate area in " << 1000 * ((double)getTickCount() - tic) / getTickFrequency() << " ms" << endl;
    cout << "area of 2D points: " << area << endl;

    cv::Rect roi(leftBottom.x, leftBottom.y, rightUp.x - leftBottom.x, rightUp.y - leftBottom.y);
    cv::Mat imageRoi = image(roi);
    imwrite("rframe-roi.jpg", imageRoi);

    Mat img_display;
    image2.copyTo(img_display);

    Mat result;
    int result_cols = image2.cols - imageRoi.cols + 1;
    int result_rows = image2.rows - imageRoi.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    //  TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED
    TemplateMatchModes match_method = TM_CCOEFF_NORMED;
    double match_threshold = 0.6;
    bool isMatched = false;

    matchTemplate(image2, imageRoi, result, match_method);

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
    {
        isMatched = minVal <= -match_threshold;
        matchLoc = minLoc;
    }
    else
    {
        isMatched = maxVal >= match_threshold;
        matchLoc = maxLoc;
    }

    if (isMatched)
    {
        string matchedFile = "rframe-matched.jpg";
        normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
        rectangle(img_display, matchLoc, Point(matchLoc.x + imageRoi.cols, matchLoc.y + imageRoi.rows), Scalar::all(0), 2, 8, 0);
        rectangle(result, matchLoc, Point(matchLoc.x + imageRoi.cols, matchLoc.y + imageRoi.rows), Scalar::all(0), 2, 8, 0);
        imwrite(matchedFile, img_display);
        cout << "pattern matched found and markered to " << matchedFile << endl;
    }
    else
    {
        cout << "no pattern matched found!" << endl;
    }
}

void testGeometry()
{
    Vector3d eAngle(1, 2, 3); // x, y, z
    Matrix3d rM = eulers2Matrix(eAngle(2), eAngle(1), eAngle(0));

    cout << "Euler angles: " << eAngle << endl;
    cout << "Rotation matrix from utility: " << rM << endl;
    cout << "Euler angles from Eigen: " << rM.eulerAngles(2, 1, 0) << endl;

    Quaterniond q = Quaterniond(rM);
    q.normalize();
    cout << "Quaterniond from Eigen: " << q.w() << endl
         << q.vec() << endl;
    cout << "Rotation matrix from Quaterniond: " << quaternion2Matrix(q) << endl;

    Quaterniond qE = eulers2Quaterniond(eAngle);
    cout << "Quaterniond from Eulers: " << qE.w() << endl
         << qE.vec() << endl;
    cout << "Euler angles from Quaterniond: " << quaternion2Eulers(q) << endl;
}

void testGZProjection()
{
    Matrix3d k;
//    k << 270, 0, 320,
//        0, 270, 240,
//        0, 0, 1;
    k << 572.4114, 0, 325.2611,
        0, 573.57043, 242.04899,
        0, 0, 1;

    Matrix3d r = eulers2Matrix(-0.242452, 0.001668, -2.093417);
    Vector3d t(-0.004284, -0.006261, 1.683763);

    vector<Point2f> tos;
    vector<Point3f> froms;
    vector<Vector3d> boxes, boxesRT;

    double x = 0.4, y = 0.3, z = 0.3;
    boxes.push_back(Vector3d(x/2, y/2, z/2));
    boxes.push_back(Vector3d(x/2, -y/2, z/2));
    boxes.push_back(Vector3d(-x/2, y/2, z/2));
    boxes.push_back(Vector3d(-x/2, -y/2, z/2));
    boxes.push_back(Vector3d(x/2, y/2, -z/2));
    boxes.push_back(Vector3d(x/2, -y/2, -z/2));
    boxes.push_back(Vector3d(-x/2, y/2, -z/2));
    boxes.push_back(Vector3d(-x/2, -y/2, -z/2));
    boxes.push_back(Vector3d(0, 0, 0));

    Matrix3d rBox = eulers2Matrix(0, 0, 0);
    Vector3d tBox(0, 1.5, 0.8);

    for (int i = 0; i < boxes.size(); i++)
    {
        Vector3d p = rBox * boxes[i] + tBox;
        boxesRT.push_back(p);

        p -= t;
        froms.push_back(Point3d(p(0), p(1), p(2)));
    }

    project3DPoints(r.inverse(), Vector3d(), k, froms, tos);

    for (int i = 0; i < tos.size(); i++)
    {
        cout << "3D point: (" << froms[i].x << "," << froms[i].y << "," << froms[i].z << ")" << endl;
        cout << "2D point: (" << (int) tos[i].x << "," << (int) tos[i].y << ")" << endl;
    }
}

void testBiasW()
{
    IMUPose imu;
    double t = 1.0 / 200.0;
    Vector3d e0 = Vector3d(-2.058788, 0.007244, -0.021635);
    Quaterniond q0 =  eulers2Quaterniond(e0);
    Matrix3d r0 = imu.r = eulers2Matrix(e0(2), e0(1), e0(0));
    Vector3d e = quaternion2Eulers(Quaterniond(imu.r));
    cout << "original pose GT: " << e0 << endl;
    cout << "original pose imu: " << e << endl;

    Vector3d w = Vector3d (-0.032302, -0.013955, -0.015225);
    imu.update(w, w, t);

    Vector3d e1 = Vector3d(-2.058949, 0.007209, -0.021538);
    Quaterniond q1 =  eulers2Quaterniond(e1);
    Matrix3d r1 = eulers2Matrix(e1(2), e1(1), e1(0));

    Matrix3d dr = r0.inverse() * r1;
    Quaterniond dq = Quaterniond(dr);
    Vector3d de = quaternion2Eulers(dq);
    Vector3d w_ = de / t;

//    Vector4d dq = Vector4d(
//                           q1.x() - q0.x(),
//                           q1.y() - q0.y(),
//                           q1.z() - q0.z(),
//                           q1.w() - q0.w()
//                           );
//    dq = dq * 2.0 / t;
//    Eigen::Matrix<double, 4, 3> Q;
//    Q << q0.w(), -q0.z(), q0.y(),
//        q0.z(), q0.w(), -q0.x(),
//        -q0.y(), q0.x(), q0.w(),
//        -q0.x(), -q0.y(), -q0.z();


//    Vector3d w_ = (Q.transpose() * Q).inverse() * Vector3d(dq(0), dq(1), dq(2));

    e = quaternion2Eulers(Quaterniond(imu.r));
    cout << "new pose GT: " << e1 << endl;
    cout << "new pose imu: " << e << endl;

    cout << "w: " << w << endl;
    cout << "w_: " << w_ << endl;
}

// test pose by drawing bounding box
void testDuck()
{
    Matrix3d _cameraK;
    ObjectInfo _object;
    string _modelCornerPath = "/Users/sherlock/Downloads/simulation/results/corner.txt";
    string _modelConfigPath = "/Users/sherlock/Downloads/simulation/results/model_config.txt";
    string _intrinsicPath = "/Users/sherlock/Downloads/simulation/results/intrinsic.txt";

    bool isNativePrj = false;
    loadCameraInfo(&_cameraK, _intrinsicPath);
    loadObjectInfo(&_object, _modelCornerPath, _modelConfigPath, &isNativePrj);
    setProjectionType(isNativePrj);

    BasicPose pose = BasicPose(eulers2Matrix(-0.002116, 0.005044, 0.009318), Vector3d(0.002411, 0.000613, -0.000839));
    pose = BasicPose(eulers2Matrix(-0.283173, -0.359864, 0.004535), Vector3d(0.129389, 0.014370, 0.084065));
    pose = BasicPose(eulers2Matrix(0, 0, 0), Vector3d(0, -0.01, 0));
    pose = BasicPose(eulers2Matrix(1.016351, 1.338304, -0.031411), Vector3d(-0.379833, -0.338180, 0.328426));
    pose = BasicPose(eulers2Matrix(1.75577069, 0.93394343, 1.04973741), Vector3d(-0.379833, -0.338180, 0.328426));
    pose = BasicPose(eulers2Matrix(1.282772, 1.015671, 2.836318), Vector3d(-0.000931, -0.024577, 0.338934));

    vector<Point2f> boxesP = getBoundingbox2DPoints(_object,_cameraK, pose);

    for (int i = 0; i < boxesP.size(); i++)
        cout << "2D point: (" <<  boxesP[i].x << "," <<  boxesP[i].y << ")" << endl;

    string srcFramePath = "/Users/sherlock/Downloads/simulation/results/frames/00000000.jpg";
    string dstFramePath = "/Users/sherlock/Downloads/simulation/results/frames/00000000-box.jpg";

    drawBoundingbox(srcFramePath, dstFramePath, pose, _object, _cameraK);
}
