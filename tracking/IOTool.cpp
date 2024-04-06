//
//  IOTool.cpp
//  FrontendTracking
//
//  Created by Lau Yo-Chung on 2022/5/28.
//  Copyright © 2022 Lau Yo-Chung. All rights reserved.
//

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "IOTool.hpp"
#include "SettingDefine.h"
#include "Utility.hpp"
#include "PCM.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

// SUV bounding box:
//    _object.boxes.push_back(Vector3d(1.07, 0.13, 0.108));
//    _object.boxes.push_back(Vector3d(1.07, -0.13, 0.108));
//    _object.boxes.push_back(Vector3d(0.93, 0.13, 0.108));
//    _object.boxes.push_back(Vector3d(0.93, -0.13, 0.108));
//    _object.boxes.push_back(Vector3d(1.07, 0.13, -0.108));
//    _object.boxes.push_back(Vector3d(1.07, -0.13, -0.108));
//    _object.boxes.push_back(Vector3d(0.93, 0.13, -0.108));
//    _object.boxes.push_back(Vector3d(0.93, -0.13, -0.108));
void loadObjectInfo (ObjectInfo *obj, std::string cornerPath, std::string configPath, bool *isNativeProj)
{
    if (!obj)
        return;;

    if (cornerPath.length() > 0)
    {
        ifstream fp; fp.open(cornerPath);
        while (1)
        {
            string line;
            getline(fp, line);
            if (line.empty() || line.length() < 3)
                break;
            else
            {
                vector<string> d(getLineSub(line, ' '));
                obj->boxes.push_back(Vector3d(atof(d[0].c_str()), atof(d[1].c_str()), atof(d[2].c_str())));
            }
        }
        fp.close();
    }

    if (configPath.length() > 0)
    {
        Vector3d rts[3] = {Vector3d(), Vector3d(), Vector3d(1, 1, 1)};
        
        ifstream fp; fp.open(configPath);
        for (int i = 0; i < 3; i++)     // R, T, S, just 3 rows expectedly in cornerPath
        {
            string line;
            getline(fp, line);
            if (line.empty() || line.length() < 3)
                break;
            else
            {
                vector<string> d(getLineSub(line, ' '));
                rts[i] = Vector3d(atof(d[0].c_str()), atof(d[1].c_str()), atof(d[2].c_str()));
            }
        }
        string line;
        getline(fp, line);
        if (!line.empty())
        {
            vector<string> d(getLineSub(line, ' '));
            *isNativeProj = atoi(d[0].c_str()) != 0;
        }
        else
            *isNativeProj = false;
        fp.close();

        obj->r = eulers2Matrix(rts[0](2), rts[0](1), rts[0](0));
        obj->t = Vector3d(rts[1](0), rts[1](1), rts[1](2));
        obj->s = Vector3d(rts[2](0), rts[2](1), rts[2](2));
        obj->transform();
    }
}

void loadCameraInfo (Eigen::Matrix3d *K, std::string intrinsicPath)
{
    if (K && intrinsicPath.length() > 0)
    {
        ifstream fp; fp.open(intrinsicPath);
        string line;
        getline(fp, line);
        if (!line.empty() && line.length() > 0)
        {
            vector<string> d(getLineSub(line, ' '));
            (*K) << atof(d[0].c_str()),  atof(d[1].c_str()),  atof(d[2].c_str()),
                    atof(d[3].c_str()),  atof(d[4].c_str()),  atof(d[5].c_str()),
                    atof(d[6].c_str()),  atof(d[7].c_str()),  atof(d[8].c_str());
        }
        fp.close();
    }
}

void drawBoundingbox(string srcFramePath, string dstFramePath, BasicPose pose, ObjectInfo object, Matrix3d cameraK, Scalar color)
{
    if (!(srcFramePath.length() > 0 && dstFramePath.length() > 0))
        return;;

    Mat frame = imread(srcFramePath, CV_LOAD_IMAGE_COLOR);
    if (frame.empty())
        return;

    vector<Point2f> boxesP = getBoundingbox2DPoints(object, cameraK, pose);
    vector<vector<Point>> contours;
    vector<Point> contour;

    for (int j = 0; j < 2; j++)
    {
        contour.clear();
        for (int i = 0; i < 4; i++)
            contour.push_back(Point(round(boxesP[j * 4 + i].x), round(boxesP[j * 4 + i].y)));
        contours.push_back(contour);
    }

    contour.clear();
    contour.push_back(Point(round(boxesP[0].x), round(boxesP[0].y)));
    contour.push_back(Point(round(boxesP[1].x), round(boxesP[1].y)));
    contour.push_back(Point(round(boxesP[5].x), round(boxesP[5].y)));
    contour.push_back(Point(round(boxesP[4].x), round(boxesP[4].y)));
    contours.push_back(contour);

    contour.clear();
    contour.push_back(Point(round(boxesP[2].x), round(boxesP[2].y)));
    contour.push_back(Point(round(boxesP[3].x), round(boxesP[3].y)));
    contour.push_back(Point(round(boxesP[7].x), round(boxesP[7].y)));
    contour.push_back(Point(round(boxesP[6].x), round(boxesP[6].y)));
    contours.push_back(contour);

 //   if ((color.val[0] == color.val[1] == color.val[2]))
        drawContours(frame, contours, -1, color, 1.5, LINE_AA);

//    for (int i = 0; i < boxesP.size(); i++)
//        for (int j = i + 1; j < boxesP.size(); j++)
//            if ((color.val[0] == color.val[1] == color.val[2]))
//                line(frame, Point(round(boxesP[i].x), round(boxesP[i].y)), Point(round(boxesP[j].x), round(boxesP[j].y)), color, 1, LINE_AA);

    imwrite(dstFramePath, frame);
}

void noiseFramePose(FrameInfo &fi, bool isAlways)
{
    double z = abs(getRandomNormal(0, 1.0));
    if (isAlways || z > _ND_Z_99){
//        cout<<"Start noising pose!" << endl;

        double bias_1 = 0.5773502691896;
        double unit = bias_1 / 100;
        fi.p = Vector3d(fi.pGT(0) + getRandomNormal(0, 1.0, 1) * 0.25 * unit,
                        fi.pGT(1) + getRandomNormal(0, 1.0, 1) * 0.25 * unit,
                        fi.pGT(2) + getRandomNormal(0, 1.0, 1) * 0.25 * unit);

        unit = M_PI * bias_1 / 180.0;
        fi.e = Vector3d(fi.eGT(0) + getRandomNormal(0, 1.0, 1) * 0.25 * unit,
                        fi.eGT(1) + getRandomNormal(0, 1.0, 1) * 0.25 * unit,
                        fi.eGT(2) + getRandomNormal(0, 1.0, 1) * 0.25 * unit);
    }
    else {
        fi.p = fi.pGT;
        fi.e = fi.eGT;
    }
    fi.r = eulers2Matrix(fi.e(2), fi.e(1), fi.e(0));
}

// fp refers to image_pose2.txt
bool readFrameInfo(std::ifstream &fp, FrameInfo &fi, SharingData sharingData, bool isFrameNeeded, bool is2ShowLog)
{
    fi.time = INFINITY;

    string line;
    getline(fp, line);

    if (line.empty() || line.length() < 3)
    {
        cout << "Out of image frames..." << endl;
        fi.status = false;
    }
    else
    {
        vector<string> d(getLineSub(line, ' '));
        fi.basePath = sharingData._baseFramePath;
        fi.baseBoxPath = sharingData._baseBoxPath;
        fi.fileName = d[0] + ".jpg";
        fi.index = atoi(d[0].c_str());
        fi.time_s = atof(d[1].c_str());
        fi.time_ms = atof(d[2].c_str()) / _MS_BASE;
        fi.time = fi.time_s + fi.time_ms;
        fi.duration = atof(d[3].c_str()) * 1000; fi.duration = fi.duration < 0? fi.period : fi.duration;
        fi.toNextTime = atof(d[4].c_str()) * 1000; fi.toNextTime = fi.toNextTime < 0? fi.period : fi.toNextTime;
        fi.status = true;

        if (d.size() > 5)
            fi.p = fi.pGT = Vector3d(atof(d[5].c_str()), atof(d[6].c_str()), atof(d[7].c_str()));

        if (d.size() > 8)
            fi.e = fi.eGT = Vector3d(atof(d[8].c_str()), atof(d[9].c_str()), atof(d[10].c_str()));

        fi.r = fi.rGT = eulers2Matrix(fi.eGT(2), fi.eGT(1), fi.eGT(0));

        if (d.size() > 11)
            fi.vGT = Vector3d(atof(d[11].c_str()), atof(d[12].c_str()), atof(d[13].c_str()));

        if (d.size() > 14)
            fi.wGT = Vector3d(atof(d[14].c_str()), atof(d[15].c_str()), atof(d[16].c_str()));

        if (isFrameNeeded && fi.time >= sharingData._startTime && fi.time <= sharingData._endTime)
        {
            string framePath = fi.basePath + fi.fileName;
            fi.frame = imread(framePath, CV_LOAD_IMAGE_GRAYSCALE);
            if (fi.frame.empty())
            {
                cout << "Cannot read frame raw data of " << framePath << endl;
                fi.status = false;
            }
        }

        if (is2ShowLog)
        {
            cout << "frame index: " << fi.index << endl;
            cout << "frame file: " << fi.fileName << endl;
            cout << "frame time: " << fi.time << endl;
            cout << "frame rotation GT: " << fi.p << endl;
            cout << "frame position GT: " << fi.r << endl;
        }
    }

    return fi.status;
}

bool readIMUInfo(std::ifstream &fp, IMUPose &imu, bool is2ShowLog)
{
    string line;
    getline(fp, line);
    vector<string> d(getLineSub(line, ' '));

    if (line.empty() || line.length() < 10)
        imu.status = false;
    else
    {
        imu.status = true;
        imu.time = imu.timeRead = atof(d[0].c_str()) + atof(d[1].c_str()) / _MS_BASE;
        imu.duration = atof(d[2].c_str()) * 1e3; imu.duration = imu.duration < 0? imu.period : imu.duration;
        imu.toNextTime = atof(d[3].c_str()) * 1e3; imu.toNextTime = imu.toNextTime < 0? imu.period : imu.toNextTime;
        imu.a = Vector3d(atof(d[4].c_str()), atof(d[5].c_str()), atof(d[6].c_str()));
        imu.w = Vector3d(atof(d[7].c_str()), atof(d[8].c_str()), atof(d[9].c_str()));

        if (is2ShowLog)
        {
            cout << "IMU time: " << imu.time << endl;
            cout << "IMU duration: " << imu.duration << endl;
            cout << "IMU toNextTime: " << imu.toNextTime << endl;
            cout << "IMU a: " << imu.a << endl;
            cout << "IMU w: " << imu.w << endl;
        }
    }

    return imu.status;
}

void loadDefaultBiasFromData(IMUPose &imu, Eigen::Vector3d a = Vector3d(0, 0, 0), Eigen::Vector3d w = Vector3d(0, 0, 0))
{
    imu.bias_a = a;
    imu.bias_w = w;

    cout << "IMUPose::bias_a: " << imu.bias_a << endl;
    cout << "IMUPose::bias_w: " << imu.bias_w << endl;
}

void loadDefaultBiasFromFile(IMUPose &imu, std:: string fileName)
{
    if (fileName.length() > 0)
    {
        ifstream fp; fp.open(fileName);
        if (fp.is_open())
        {
            Vector3d data[] = {Vector3d(0, 0, 0), Vector3d(0, 0, 0)};       // {bias_a, bias_w}
            string line;

            for (int i = 0; i < 2; i++)
            {
                getline(fp, line);
                vector<string> d(getLineSub(line, ' '));

                if (line.empty() || line.length() < 3)
                    break;;

                data[i] = Vector3d(atof(d[0].c_str()), atof(d[1].c_str()), atof(d[2].c_str()));
            }

            loadDefaultBiasFromData(imu, data[0], data[1]);
            fp.close();
        }
    }

}

void addDurationAndNextTime(std::string srcPath, std::string dstPath, int timeIndex)
{
    if (!(srcPath.length() > 0 && dstPath.length() > 0))
        return;;

    cout << "Start adding duration and next time from " << srcPath << endl;

    remove(dstPath.c_str());

    ifstream inFile; inFile.open(srcPath);
    ofstream outFile; outFile.open(dstPath);
    string line, lineNext;
    double lastTime = -1, durationNow = 0;

    while (1)
    {
        getline(inFile, lineNext);

        if (lastTime < 0)   // 1st
        {
            if (lineNext.empty() || lineNext.length() < 5)
            {
                cout << "Out of data..." << endl;
                break;;
            }
            else
            {
                vector<string> dNext(getLineSub(lineNext, ' '));
                lastTime = 1.0 + atof(dNext[timeIndex].c_str()) + atof(dNext[timeIndex + 1].c_str()) / _MS_BASE;
                line = lineNext;
            }

            continue;;
        }

        if (lineNext.empty() || lineNext.length() < 5)
        {
            cout << "Out of data..." << endl;

            vector<string> dNow(getLineSub(line, ' '));
            double timeNow = atof(dNow[timeIndex].c_str()) + atof(dNow[timeIndex + 1].c_str()) / _MS_BASE;
            durationNow = timeNow - lastTime;

            for (int i = 0; i < timeIndex + 2; i++)
                outFile << dNow[i] << " ";
            outFile << fixed << setprecision(6) << durationNow << " " << -1;

            for (int i = timeIndex + 2; i < dNow.size(); i++)
                outFile << " " << dNow[i];
            outFile << endl;

            break;;
        }
        else
        {
            vector<string> dNow(getLineSub(line, ' '));
            vector<string> dNext(getLineSub(lineNext, ' '));

            double timeNow = atof(dNow[timeIndex].c_str()) + atof(dNow[timeIndex + 1].c_str()) / _MS_BASE;
            double timeNext = atof(dNext[timeIndex].c_str()) + atof(dNext[timeIndex + 1].c_str()) / _MS_BASE;
            durationNow = timeNow - lastTime;

            for (int i = 0; i < timeIndex + 2; i++)
                outFile << dNow[i] << " ";
            outFile << fixed << setprecision(6) << durationNow << " " << timeNext - timeNow;

            for (int i = timeIndex + 2; i < dNow.size(); i++)
                outFile << " " << dNow[i];
            outFile << endl;

            lastTime = timeNow;
            line = lineNext;
        }
    }

    inFile.close();
    outFile.close();

    cout << "Finished adding duration and next time to " << dstPath << endl;
}
