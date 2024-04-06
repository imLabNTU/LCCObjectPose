#include <thread>
#include <exception>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataStructure.h"
#include "Utility.hpp"
#include "IOTool.hpp"
#include "PCM.hpp"
#include "IMUPose.hpp"
#include "test.hpp"
#include "Setting.h"

using namespace std;
using namespace cv;
using namespace Eigen;

//#define _USING_C_SLEEP
#ifdef _USING_C_SLEEP
    #define _mySleep(ns)    {if ((ns) >= 0) this_thread::sleep_for(chrono::nanoseconds(ns));}
#else
    #define _mySleep(ns)    { if ((ns) >= 0) { \
                                auto spinStart = chrono::high_resolution_clock::now(); \
                                int delayTime = (int) (ns); \
                                while ((chrono::high_resolution_clock::now() - spinStart).count() < delayTime);}}
#endif

//#define _TEST_PCM
#define _TEST_IMU
//#define _TEST_EIGEN
//#define _TEST_FILE
//#define _TEST_FRAMES
//#define _TEST_GEOMETRY
//#define _TEST_GZPRJ
//#define _TEST_BIASW
//#define _TEST_DUCK

//#define _SHOW_LOG
//#define _SHOW_PP_TIME
//#define _SHOW_SLEEP_TIME

SharingData _sharingData;

ofstream _trackingPoseFP;

bool _is2TestTHR2D = false;

int _backCountK = 1;
int _frontCountL = 1;

Matrix3d _cameraK;
ObjectInfo _object;

mutex _mu_imuPose;
IMUPose _imuPose, _imuPoseTwin;
BasicPose _lastFramePose;
bool _isByRK4 = false;

mutex _mu_nowFrame;
FrameInfo _nowFrame;

mutex _mu_trackingResult;
struct TrackingResult
{
    TrackingResult() {
        clear();
    };

    void save (FrameInfo frame, IMUPose imu, bool isTracking) {
        _mu_trackingResult.lock();
        frames.push_back(frame); imus.push_back(imu); isTrackings.push_back(isTracking);
        _mu_trackingResult.unlock();
    };

    void clear() {
        frames.clear(); imus.clear(); isTrackings.clear();
    };

    void pop()
    {
        if (!isWithData())
        {
            cout << "Warning! No data to pup up or the status of TrackingResult is not synchronous!" << endl;
            cout << "(framesEmpty, imusEmpty, isTrackingsEmpty): (" << frames.empty() << " ," << imus.empty() << " ," << isTrackings.empty() << ")" << endl;
            return;;
        }

        _mu_trackingResult.lock();
        if (!frames.empty())
            frames.erase(frames.begin());

        if (!imus.empty())
            imus.erase(imus.begin());

        if (!isTrackings.empty())
            isTrackings.erase(isTrackings.begin());
        _mu_trackingResult.unlock();
    }

    bool isEmpty() {
        return frames.empty() && imus.empty() && isTrackings.empty();
    }

    bool isWithData() {
        return !frames.empty() && !imus.empty() && !isTrackings.empty();
    }

    vector<FrameInfo> frames;
    vector<IMUPose> imus;
    vector<bool> isTrackings;
};
TrackingResult _trackingResult;

enum SystemStatus
{
    SystemStatus_Ini = 0,
    SystemStatus_0_Send,
    SystemStatus_0_Received,
    SystemStatus_1_Send,
    SystemStatus_1_Received,
};
SystemStatus _status = SystemStatus_Ini;
mutex _mu_status;

long _1stFrameIndex = -1;

void saveTrackingPose(FrameInfo frame, IMUPose imu, bool isTracking = true)
{
    Vector3d p = imu.p;
    Vector3d e = quaternion2Eulers(Quaterniond(imu.r));

    Vector3d gp = frame.p;
    Vector3d ge = frame.e;

#ifdef _SHOW_LOG
    cout << "current frame of " << frame.index + _1stFrameIndex << " at " << frame.time << endl;
    cout << "current velocity: " << "(" << imu.v(0) << ", " << imu.v(1) << ", " << imu.v(2) << ")" << endl;
    cout << "correct velocity: " << "(" << frame.vGT(0) << ", " << frame.vGT(1) << ", " << frame.vGT(2) << ")" << endl;
#endif

    _trackingPoseFP << frame.fileName << fixed << setprecision(6)
                    << " " << p(0) << " " << p(1) << " " << p(2)
                    << " " << e(0) << " " << e(1) << " " << e(2)
                    << " " << gp(0) << " " << gp(1) << " " << gp(2)
                    << " " << ge(0) << " " << ge(1) << " " << ge(2)
                    << " " << (isTracking? 1 : 0)
                    << endl;
}

bool _isTrackingEnd = false;
void saveTrackingPoseRoutine()
{
    try {
        while (!(_isTrackingEnd && _trackingResult.isEmpty()))
        {
            if (_trackingResult.isWithData())
            {
                saveTrackingPose(_trackingResult.frames[0], _trackingResult.imus[0], _trackingResult.isTrackings[0]);
                _trackingResult.pop();
            }
        }
    } catch (exception& e) {
        cout << "catch error in saveTrackingPoseRoutine with: " << e.what() << endl;
    }
}

void postProcessing()
{
    cout << "Start doing post-processing..." << endl;

    system(("rm -r -f " + _sharingData._baseBoxPath).c_str());
    system(("mkdir " + _sharingData._baseBoxPath).c_str());

    remove(_sharingData._trackingErrorPath.c_str());
    ofstream ofp;
    ofp.open(_sharingData._trackingErrorPath);

    ifstream fp;
    fp.open(_sharingData._trackingPosePath);

    while (1)
    {
        string line;
        getline(fp, line);
        if (line.empty() || line.length() < 10)
            break;
        else
        {
            vector<string> d(getLineSub(line, ' '));
            Vector3d e = Vector3d(atof(d[4].c_str()), atof(d[5].c_str()), atof(d[6].c_str()));
            BasicPose pose = BasicPose(eulers2Matrix(e(2), e(1), e(0)), Vector3d(atof(d[1].c_str()), atof(d[2].c_str()), atof(d[3].c_str())));
            string indexStr = d[0].substr(0, d[0].length() - 4);        // d[0] = "*.jpg"
            string fileName = itosWith0(atoi(indexStr.c_str()) + _1stFrameIndex) + ".jpg";
            string dstPath = _sharingData._baseBoxPath + fileName;
            if (dstPath.compare(dstPath.length() - 4, dstPath.length(), ".jpg"))
                dstPath += ".jpg";

            drawBoundingbox(_sharingData._baseFramePath + fileName, dstPath, pose, _object, _cameraK, _sharingData._poseColorSF);

            Vector3d ge = Vector3d(atof(d[10].c_str()), atof(d[11].c_str()), atof(d[12].c_str()));
            BasicPose poseGT = BasicPose(eulers2Matrix(ge(2), ge(1), ge(0)), Vector3d(atof(d[7].c_str()), atof(d[8].c_str()), atof(d[9].c_str())));
            drawBoundingbox(dstPath, dstPath, poseGT, _object, _cameraK, _sharingData._poseColorGT);

            Vector3d de = ge - e, dp = poseGT.t - pose.t;
            for (int i = 0; i < 3; i++)
            {
                de(i) = de(i) < 0? -de(i) : de(i);
                de(i) = de(i) > M_PI? 2.0 * M_PI - de(i) : de(i);
                dp(i) = dp(i) < 0? -dp(i) : dp(i);
            }

            double dis = getBoundingbox2DPointsDiff(&_object, _cameraK, pose, poseGT);
//            cout << "dis: " << dis << endl;

            ofp << fileName.substr(0, fileName.length() - 4) << fixed << setprecision(6) << " " << dis
//            ofp << fileName << fixed << setprecision(6) << " " << dis
                << " " << dp(0) << " " << dp(1) << " " << dp(2) << " " << dp.norm()
                << " " << de(0) << " " << de(1) << " " << de(2) << " " << de.norm()
                << " " << d[13]
                << endl;
        }
    }

    ofp.close();
    fp.close();

    cout << "Finished doing post-processing!" << endl;
}

void getFrameOrder(vector<char> &serials)
{
    serials.clear();

    if (_frontCountL == 0)
    {
        for (int i = 0; i < _sharingData._frameRate; i++)
            serials.push_back('b');

        return;
    }

    if (_backCountK == 1)
    {
        serials.push_back('b');
        for (int i = 1; i < _sharingData._frameRate; i++)
            serials.push_back('f');

        return;
    }

    if (_frontCountL == _backCountK)
    {
        for (int i = 0; i < _sharingData._frameRate; i++)
            serials.push_back(i % 2 == 0 ? 'b' : 'f');
    }
    else
    {
        vector<vector<char>> serials2;
        int base = MIN(_frontCountL, _backCountK);
        char first = _frontCountL > _backCountK ? 'b' : 'f';
        char later = _frontCountL > _backCountK ? 'f' : 'b';

        for (int i = 0; i < base; i++)
        {
            vector<char> s;
            s.push_back(first);
            serials2.push_back(s);
        }

        for (int i = base; i < _sharingData._frameRate; i++)
            serials2[i % base].push_back(later);

        if (first == 'b')
        {
            for (int i = 0; i < serials2.size(); i++)
                for (int j = 0; j < serials2[i].size(); j++)
                    serials.push_back(serials2[i][j]);
        }
        else
        {
            for (int i = 0; i < serials2.size(); i++)
                for (int j = (int)serials2[i].size() - 1; j >= 0; j--)
                    serials.push_back(serials2[i][j]);
        }
    }
}

void cullFrames()
{
    ifstream fp;
    fp.open(_sharingData._imagePose2Path);

    while (1)
    {
        FrameInfo fi;
        if (!readFrameInfo(fp, fi, _sharingData, true))
            break;

        if (fi.time < _sharingData._startTime || fi.time > _sharingData._endTime)
        {
            string path = fi.basePath + fi.fileName;
            remove(path.c_str());
        }
    };

    fp.close();
}

// no need to do mutex because of the seqential processing
bool DFCM(int frameRate, int imuRate, vector<char> &serials)
{
    srand((unsigned) time(NULL));

    remove(_sharingData._dfcmPath.c_str());
    ofstream dfcmFP;
    dfcmFP.open(_sharingData._dfcmPath);

    _sharingData._timeBack = _sharingData._baseTimeBack + _sharingData._extraDelay / 1000.0;

    _nowFrame.rate = _sharingData._frameRate = frameRate;
    _nowFrame.period = 1000.0 / ((double) _nowFrame.rate);

    _imuPose.rate = _sharingData._imuRate = imuRate;
    _imuPose.period = 1000.0 / ((double) _imuPose.rate);

    bool isNativePrj = false;
    loadCameraInfo(&_cameraK, _sharingData._intrinsicPath);
    loadObjectInfo(&_object, _sharingData._modelCornerPath, _sharingData._modelConfigPath, &isNativePrj);
    setProjectionType(isNativePrj);

    int f = _sharingData._frameRate;
    int fb = int(1.0 / _sharingData._timeBack);
    int ff = int(1.0 + (1.0 - _sharingData._timeBack) / _sharingData._timeFront);

    bool isSatisfied = false;
    if (f <= fb)
    {
        _backCountK = _sharingData._frameRate;
        _frontCountL = 0;
        isSatisfied = true;
    }
    else
    {
        if (f <= ff)
        {
            _backCountK = int((1.0 - f * _sharingData._timeFront) / (_sharingData._timeBack - _sharingData._timeFront));
            _frontCountL = f - _backCountK;
            isSatisfied = true;
        }
    }

    dfcmFP << "The tracking processing folder: " << _sharingData._baseSimPath << endl;
    dfcmFP << "The expected frame rate: " << _sharingData._frameRate << " (FPS)" << endl;
    dfcmFP << "The IMU rate: " << _sharingData._imuRate << " (HZ)" << endl;
    dfcmFP << "The backend tracking time (tb): " << _sharingData._timeBack * 1000 << " (ms)" << endl;
    dfcmFP << "The frontend tracking time (tf): " << _sharingData._timeFront * 1000 << " (ms)" << endl;
    if (isSatisfied)
    {
        dfcmFP << "The expected backend tracking count (k): " << _backCountK << endl;
        dfcmFP << "The expected frontend tracking count (l): " << _frontCountL << endl;

        getFrameOrder(serials);
        dfcmFP << "The frame processing order:" << endl;
        for (vector<char>::iterator itr = serials.begin(); itr != serials.end(); itr++)
            dfcmFP << *itr << " ";
        dfcmFP << endl;
    }
    else
        dfcmFP << "The expected frame rate cannot be satisfied..." << endl;

    dfcmFP.close();

    ifstream fp; fp.open(_sharingData._dfcmPath);
    string line;
    while (getline(fp, line))
        cout << line << endl;
    fp.close();

    return isSatisfied;
}

// no need to do mutex because of the seqential processing
bool init(int frameRate, int imuRate, vector<char> &serials, bool is2ShowLog = true)
{
    srand((unsigned) time(NULL));

    remove(_sharingData._dfcmPath.c_str());
    ofstream dfcmFP;
    dfcmFP.open(_sharingData._dfcmPath);

    _sharingData._timeBack = _sharingData._baseTimeBack + _sharingData._extraDelay / 1000.0;

    _nowFrame.rate = _sharingData._frameRate = frameRate;
    _nowFrame.period = 1000.0 / ((double) _nowFrame.rate);

    _imuPose.rate = _sharingData._imuRate = imuRate;
    _imuPose.period = 1000.0 / ((double) _imuPose.rate);
    loadDefaultBiasFromData(_imuPose, Vector3d(0, 0, 0), Vector3d(0, 0, 0));
    loadDefaultBiasFromFile(_imuPose, _sharingData._biasPath);

    _sharingData._timeFront = 1.0 / _sharingData._imuRate;

    bool isNativePrj = false;
    loadCameraInfo(&_cameraK, _sharingData._intrinsicPath);
    loadObjectInfo(&_object, _sharingData._modelCornerPath, _sharingData._modelConfigPath, &isNativePrj);
    setProjectionType(isNativePrj);

    dfcmFP << "The tracking processing folder: " << _sharingData._baseSimPath << endl;
    dfcmFP << "The expected frame rate: " << _sharingData._frameRate << " (FPS)" << endl;
    dfcmFP << "The IMU rate: " << _sharingData._imuRate << " (HZ)" << endl;
    dfcmFP << "The backend tracking time (tb): " << _sharingData._timeBack * 1000 << " (ms)" << endl;
    dfcmFP << "The frontend tracking time (tf): " << _sharingData._timeFront * 1000 << " (ms)" << endl;

    bool isSatisfied = (_sharingData._timeBack <= 1000 && _sharingData._frameRate <= _sharingData._imuRate);
    if (!isSatisfied)
        dfcmFP << "The expected frame rate cannot be satisfied..." << endl;

    dfcmFP.close();

    if (is2ShowLog)
    {
        ifstream fp; fp.open(_sharingData._dfcmPath);
        string line;
        while (getline(fp, line))
            cout << line << endl;
        fp.close();
    }

    return isSatisfied;
}

// mutext in caller
double _rectfyTime = -1.0;
bool _isVCorrected = false;
FrameInfo _lastFrameInfo;
void updatePoseFromFrame(FrameInfo frame, IMUPose imuPose)
{
    #define _local_diff_thrd                (0.0)
    #define _local_bias_divider_a           (1)
    #define _local_bias_divider_w           (1)

    bool is2Avg = !(_rectfyTime < 0);

    if (_sharingData._isBSCM && _status >= SystemStatus_0_Received)
    {
        double dis = 10;
//        dis = getBoundingbox2DPointsDiff(&_object, _cameraK, BasicPose(imuPose.r, imuPose.p), BasicPose(frame.r, frame.p));
//        cout << frame.index << ": bounding box error: " << dis << endl;

        if (dis >= _local_diff_thrd)
        {
            _imuPose.r = frame.r;
            _imuPose.p = frame.p;

            double time = _rectfyTime < 0? frame.time - frame._time : frame.time - _rectfyTime;
            if (time > 0)
            {
                Vector3d dp = (frame.p - imuPose.p);
                Vector3d dv = dp / time;
                Vector3d a = dv / time;

                Matrix3d dr = imuPose.r.inverse() * frame.r;
                Quaterniond dq = Quaterniond(dr);
                Vector3d de = quaternion2Eulers(dq);
                Vector3d w = de / time;

                if (!_sharingData._isBSCMOnlyV && _status >= SystemStatus_1_Received && frame.time >= _sharingData._biasCorrectStartTime)
                {
//                    if ( (a.norm() < imuPose.bias_a.norm() * 0.5) && (a.norm() >= imuPose.bias_a.norm() * 0.1))
                    if (_isVCorrected)
                    {
                        _imuPose.bias_a = imuPose.bias_a + a / _local_bias_divider_a;
                        _imuPose.bias_w = imuPose.bias_w + w / _local_bias_divider_w;

//                        for (int i = 0; i < 3; i++)
//                            _imuPose.bias_w(i) = abs(w(i)) >= _W_BIAS_THRESHOLD? imuPose.bias_w(i) + w(i) : imuPose.bias_w(i);
                    }
                }

                _imuPose._v = imuPose._v + dv;
                _imuPose.v = imuPose.v + dv;
                _isVCorrected = true;

                _rectfyTime = frame.time;
            }
            else
                cout << "error! imu runs too fast!" << endl;
        }
        else
        {
            _isVCorrected = false;
            _imuPose.clearAccumulation();
        }
    }
    else
    {
        _imuPose.r = frame.r;
        _imuPose.p = frame.p;

        _imuPose._v = imuPose._v;
        _imuPose.v = imuPose.v;
        
//        if (_status == SystemStatus_1_Send)
//        {
//            double time = frame.time - frame._time;
//            if (time > 0)
//            {
//                _imuPose.v = _imuPose._v = (frame.p - frame._p) / time;
//            }
//        }
    }

    if (1 && is2Avg && _sharingData._isBSCM)
    {
        double t = (frame.time - _lastFrameInfo.time);
        Vector3d v = (frame.p - _lastFrameInfo.p) / t;
        Vector3d dv = v - _imuPose.v;
        Vector3d a = dv / t;

        _imuPose._v += dv;
        _imuPose.v += dv;

        _imuPose.bias_a = imuPose.bias_a + a / _local_bias_divider_a;
    }
    _lastFrameInfo = frame;

#ifdef _SHOW_LOG
    cout << "Finished backend tracking with V (GT): " << frame.vGT << " at frame: " << frame.index + _1stFrameIndex << endl;
    cout << "Reupdate pose after backend tracking expectedly from: " << frame.time << " to " << frame.time + _timeBack << endl;

    if (_imuPose.whens.size() > 0)
    {
        cout << "The number of imu data kept: " << _imuPose.whens.size() << endl;
        cout << "imu data time from " << _imuPose.whens[0] << " to " << _imuPose.whens[_imuPose.whens.size() - 1] << endl;
    }
    else
        cout << "no imu data kept during backend time!" << endl;
#endif

    _imuPose.time = -1;

    try {
        _imuPose.update(_isByRK4);
    } catch (exception& e) {
        cout << "catch error in doing '_imuPose.update(_isByRK4)' in updatePoseFromFrame with: " << e.what() << endl;
    }

#ifdef _SHOW_LOG
    cout << "imu moving v after backend pose correcting: " << "(" << _imuPose.v(0) << ", " << _imuPose.v(1) << ", " << _imuPose.v(2) << ")" << " at " << _imuPose.time << endl;
#endif

    Quaterniond qn = Quaterniond (_imuPose.r);
    _imuPose.q(0) = qn.w(); _imuPose.q(1) = qn.x(); _imuPose.q(2) = qn.y(); _imuPose.q(3) = qn.z();
//    cout << "Finished backend tracking of frame " << frame.index << endl;
}

bool _isNoMoreFrames = false;
void updateIMU(ifstream *fp, double timeBound = -1)
{
    ifstream *imuMetaFile;
    int toSleep = 1;

    if (fp)
    {
        toSleep = 0;
        imuMetaFile = fp;
    }
    else
    {
        imuMetaFile = new ifstream();
        imuMetaFile->open(_sharingData._imu2Path);
    }

    long imuUpdatingCount = 0;
    double imuTrackingTime = 0, imuTrackingTimeMax = 0;
    auto timeStart = chrono::high_resolution_clock::now();

    _mySleep(toSleep * ((int) (_imuPose.period * 1e6)))

    try {
        while (!_isNoMoreFrames)
        {
            timeStart = chrono::high_resolution_clock::now();

            double t2 = timeBound >= 0? timeBound : _nowFrame.time + _nowFrame.toNextTime / 1e3;
            double t1 = _imuPose.timeRead + _imuPose.toNextTime / 1e3;
            if (!(t1 < t2 && fabs(t1 - t2) > 1e-4))
            {
                if (fp)
                    break;
                else
                    continue;
            }

            string line; getline(*imuMetaFile, line);
            if (line.empty() || line.length() < 10)
            {
                cout << "Out of IMU data..." << endl;
                break;
            }
            else
            {
                vector<string> d(getLineSub(line, ' '));
                chrono::duration<double, nano> elapsed = chrono::high_resolution_clock::now() - timeStart;
                double ioTime = elapsed.count() / 1e6;
                auto timeEnd = chrono::high_resolution_clock::now();

                _imuPose.duration = atof(d[2].c_str()) * 1000; _imuPose.duration = _imuPose.duration < 0? _imuPose.period : _imuPose.duration;
                _imuPose.toNextTime = atof(d[3].c_str()) * 1000; _imuPose.toNextTime = _imuPose.toNextTime < 0? _imuPose.period : _imuPose.toNextTime;

                double time = atof(d[0].c_str()) + atof(d[1].c_str()) / _MS_BASE;
                Vector3d a(atof(d[4].c_str()), atof(d[5].c_str()), atof(d[6].c_str()));
                Vector3d w(atof(d[7].c_str()), atof(d[8].c_str()), atof(d[9].c_str()));
                double t = _imuPose.duration / 1000;

                try {
                    _imuPose.saveAccumulation(w, a, t, time);
                } catch (exception& e) {
                    cout << "catch error in doing '_imuPose.saveAccumulation(w, a, t, time)' in updateIMU with: " << e.what() << endl;
                }

                _imuPose.timeRead = time;

                if (_mu_imuPose.try_lock()) {
                    if (time < _sharingData._startTime)
                    {
                        _imuPose.clearAccumulation();
                        _mu_imuPose.unlock();
                        continue;
                    }

                    elapsed = chrono::high_resolution_clock::now() - timeStart;
                    ioTime = elapsed.count() / 1e6;
                    timeEnd = chrono::high_resolution_clock::now();

                    try {
                        _imuPose.update(_isByRK4, true);
                        _imuPoseTwin.r = _imuPose.r; _imuPoseTwin.p = _imuPose.p;

                        if (_isByRK4)
                            _imuPose.rk4(w, a, t);
                        else
                            _imuPose.update(w, a, t);
                    } catch (exception& e) {
                        cout << "catch error in doing single IMU data updating in updateIMU with: " << e.what() << endl;
                    }

                    _imuPose.time = time;

    #ifdef _SHOW_LOG
                    cout << "   imu moving v: " << "(" << _imuPose.v(0) << ", " << _imuPose.v(1) << ", " << _imuPose.v(2) << ")"
                        << " at " << _imuPose.time << endl;
    #endif
                    _mu_imuPose.unlock();

    //                double duration = (chrono::high_resolution_clock::now() - timeEnd).count() / 1e6;
    //                imuTrackingTimeMax = MAX(imuTrackingTimeMax, duration);
    //                imuTrackingTime += duration;
    //                imuUpdatingCount++;
    //                cout << "IMU pose updating time:" << duration << " ms" << endl;
    //                cout << "Max IMU pose updating time:" << imuTrackingTimeMax << " ms" << endl;
    //                cout << "(AVG) IMU pose updating time: " << imuTrackingTime / imuUpdatingCount << " ms" << endl;

                }
                else
                {
                    try {
                        _imuPose.saveLost(w, a, t, time);
                    } catch (exception& e) {
                        cout << "catch error in doing '_imuPose.saveLost(w, a, t, time)' in updateIMU with: " << e.what() << endl;
                    }

    //                cout << "   pass this imu pose updating at time: " << atof(d[0].c_str()) + atof(d[1].c_str()) / _MS_BASE << endl;
                }

                elapsed = chrono::high_resolution_clock::now() - timeEnd;
    //            cout << "imu updating time spent: " << elapsed.count() * _timeRatio / 1e6 << " (ms)" << endl;
                if (ioTime + elapsed.count() * _sharingData._timeRatio / 1e6 > _imuPose.toNextTime)
                {
                    cout << "Warning! Time spent in updating IMU is larger than 1 / imuRate!" << endl;
    //                cout << "IO time: " << ioTime << endl;
                }
                else
                {
                    _mySleep(toSleep * ((int) (_imuPose.toNextTime * 1e6 - (chrono::high_resolution_clock::now() - timeStart).count())))

    #ifdef _SHOW_SLEEP_TIME
                    cout << "imu updating time: " << (chrono::high_resolution_clock::now() - timeStart).count() / 1e6 << " (ms)" << endl;
    #endif
                }
            }
        };
    } catch (exception& e) {
        cout << "catch error in updateIMU with: " << e.what() << endl;
    }

    if (!fp)
    {
        imuMetaFile->close();
        delete imuMetaFile;
    }
}

bool _isBackendTracking = false;
mutex _mu_isBackendTracking;
void backendTracking(bool isSeq, bool isFrameUpdate, bool isPoseUpdate)
{
    long count = 0;
    double ppt = 0;
    static FrameInfo frame;
    static IMUPose imu;

    try {
        while (!_isNoMoreFrames)
        {
            auto timeStart = chrono::high_resolution_clock::now();

            if(!_isBackendTracking)
            {
                if (isSeq)
                    break;
                else
                    continue;
            }

            _sharingData._timeBack = _sharingData._isFixedExtraDelay? _sharingData._baseTimeBack + _sharingData._extraDelay / 1e3 : _sharingData._baseTimeBack + (double) getRandomUnifrom(0, _sharingData._extraDelay) / 1e3 ;
    //        cout << "current backend delay: " << _timeBack << " seconds" << endl;

            if (isFrameUpdate)
            {
                _mu_status.lock();
                if (_status < SystemStatus_1_Received)
                {
                    if (_status == SystemStatus_Ini)
                        _status = SystemStatus_0_Send;
                    else if (_status == SystemStatus_0_Received)
                        _status = SystemStatus_1_Send;
                    else
                        cout <<"SystemStatus Error" << endl;
                }
                _mu_status.unlock();

    #ifdef _SHOW_LOG
                cout << "start backend tracking at frame: " << _nowFrame.index + _1stFrameIndex << endl;
    #endif

                _mu_nowFrame.lock();
                if (_status == SystemStatus_0_Send)
                {
                    _nowFrame._p = _nowFrame.p;
                    _nowFrame._time = _nowFrame.time;
                }
                else
                {
                    _nowFrame._p = frame.p;
                    _nowFrame._time = frame.time;
                }

                frame = _nowFrame;
                _mu_nowFrame.unlock();

                _imuPoseTwin.r = _imuPose.r; _imuPoseTwin.p = _imuPose.p;
                _mu_imuPose.lock();

                try {
                    _imuPose.update(_isByRK4, true);
                    _imuPose.clearAccumulation();
                    imu = _imuPoseTwin = _imuPose;
                } catch (exception& e) {
                    cout << "catch error in doing backend pose updating in backendTracking with: " << e.what() << endl;
                }

                _mu_imuPose.unlock();
            }

            if (isPoseUpdate && _status >= SystemStatus_0_Send)
            {
                int toSleep = isSeq? 0 : 1;
                _mySleep(toSleep * ((int (MAX((_sharingData._timeBack * 1e9 - (chrono::high_resolution_clock::now() - timeStart).count()), 0)))))
    #ifdef _SHOW_SLEEP_TIME
                cout << "backend networking time: " << (chrono::high_resolution_clock::now() - timeStart).count() / 1e6 << " (ms)" << endl;
    #endif

                _imuPoseTwin.r = _imuPose.r; _imuPoseTwin.p = _imuPose.p;
                _mu_imuPose.lock();

                auto timeEnd = chrono::high_resolution_clock::now();
                updatePoseFromFrame(frame, imu);
                double duration = (chrono::high_resolution_clock::now() - timeEnd).count();

    #ifdef _SHOW_PP_TIME
                cout << "Time spent in updatePoseFromFrame: " << duration * _timeRatio / 1e6 << " (ms)" << endl;
    #endif

                if (duration * _sharingData._timeRatio > imu.period * 1e6)
                {
                    cout << "Warning! Time spent in updatePoseFromFrame is larger than imu sample time: "
                        << duration * _sharingData._timeRatio / 1e6 << " ms"  << endl;
                }

                _mySleep(toSleep * ((int) (duration * (_sharingData._timeRatio - 1))))

    #ifdef _SHOW_PP_TIME
                ppt += (_sharingData._timeRatio * duration / 1e6);
                cout << "The average processing time of updatePoseFromFrame: " << ppt / ++count << endl;
    #endif
                _mu_imuPose.unlock();

                if (_status < SystemStatus_1_Received)
                {
                    _mu_status.lock();
                    if (_status == SystemStatus_0_Send)
                        _status = SystemStatus_0_Received;
                    else if (_status == SystemStatus_1_Send)
                        _status = SystemStatus_1_Received;
                    else
                        cout <<"SystemStatus Error" << endl;
                    _mu_status.unlock();
                }
            }

            _mu_isBackendTracking.lock();
            _isBackendTracking = false;
            _mu_isBackendTracking.unlock();
        };
    } catch (exception& e) {
        cout << "catch error in backendTracking with: " << e.what() << endl;
    }
}

//#define _PCM_WITH_GT
//#define _SHOW_FRONTEND_PRC_TIME
bool _isFrontendTracking = false;
mutex _mu_isFrontendTracking;
void frontendTracking(bool isSeq = false)
{
#ifdef _SHOW_FRONTEND_PRC_TIME
    static long _frontendTimes = 0;
    static double _frontendTime = 0;
#endif

    FrameInfo frame;
    IMUPose imu;

    try {
        while (!_isNoMoreFrames)
        {
            if(!_isFrontendTracking)
                continue;;

            frame = _nowFrame;

            if (_mu_imuPose.try_lock())
            {
                try {
                    imu = _imuPose;
                } catch (exception& e) {
                    cout << "catch error in doing _imuPose assignment in frontendTracking with: " << e.what() << endl;
                }

                _mu_imuPose.unlock();
            }
            else
                imu = _imuPoseTwin;

            if (_status >= SystemStatus_1_Received)
            {
                auto timeStart = chrono::high_resolution_clock::now();

    #ifdef _PCM_WITH_GT
                IMUPose imu2 = imu; imu2.r = frame.r; imu2.p = frame.p;
                bool status = doPCM(_object, _cameraK, frame, imu2, _lastFramePose);
                _lastFramePose = BasicPose(imu2.r, imu2.p);
    #else
                bool status = doPCM(_object, _cameraK, frame, imu, _lastFramePose);
                _lastFramePose = BasicPose(imu.r, imu.p);
    #endif

                chrono::duration<double, nano> elapsed = chrono::high_resolution_clock::now() - timeStart;

    #ifdef _SHOW_FRONTEND_PRC_TIME
                _frontendTime += (elapsed.count() * _timeRatio); _frontendTimes++;
                cout << "The frontend processing time (ms): " << _frontendTime / 1e6 / _frontendTimes << endl;
    #endif
                _mySleep(((int) (elapsed.count() * (_sharingData._timeRatio - 1))))

                try {
                    if(isSeq)
                        saveTrackingPose(frame, imu, status);
                    else
                    {
                       // cout << "frame saved: " << frame.index << endl;
                        _trackingResult.save(frame, imu, status);
                    }
                } catch (exception& e) {
                    cout << "catch error in saving tracking results in frontendTracking with: " << e.what() << endl;
                }

    //            cout << "Frontend tracking of frame " << frame.index << ": " << status << endl;
            }

            _mu_isFrontendTracking.lock();
            _isFrontendTracking = false;
            _mu_isFrontendTracking.unlock();

            if (isSeq)
                break;
        };
    } catch (exception& e) {
        cout << "catch error in frontendTracking with: " << e.what() << endl;
    }
}

void showFramePoseInfo()
{
#ifdef _SHOW_LOG
    if (_status >= SystemStatus_1_Received)
    {
        cout << "The pose of (" << _nowFrame.index << ") frame (around at " << _nowFrame.time << " s)..." << endl;
        cout << "IMU Time: " << _imuPose.time << " s" << endl;
//        cout << " GT rotation:" << endl << fixed << setprecision(6) << _nowFrame.r << endl;
//        cout << " IMU rotation:" << endl << fixed << setprecision(6) << _imuPose.r << endl;
//        cout << " GT translation:" << endl << fixed << setprecision(6) << _nowFrame.p << endl;
//        cout << " IMU translation:" << endl << fixed << setprecision(6) << _imuPose.p << endl;
        cout << " IMU translation diff:" << endl << fixed << setprecision(6) << (_imuPose.p - _nowFrame.p) << endl;
    }
#endif
}

void updateFrame(vector<char> serials)
{
    ifstream frameMetaFile;
    frameMetaFile.open(_sharingData._imagePose2Path);

    _nowFrame.frame = Mat::zeros(640, 480, CV_64FC1);

    _mySleep(((int) (_nowFrame.period * 1e6)))

    try {
        while (1)
        {
            auto timeStart = chrono::high_resolution_clock::now();

            double t2 = _nowFrame.time + _nowFrame.toNextTime / 1e3;
            double t1 = _imuPose.timeRead + _imuPose.toNextTime / 1e3;
            if (t1 < t2 && fabs(t1 - t2) > 1e-4)
                continue;

            _mu_nowFrame.lock();
            readFrameInfo(frameMetaFile, _nowFrame, _sharingData, true);

            if (_sharingData._isGTPoseBias)
                noiseFramePose(_nowFrame, true);

            _mu_nowFrame.unlock();

            if (!_nowFrame.status || _nowFrame.time > _sharingData._endTime)
            {
                cout << "There are no more frames left, exit..." << endl;
                _isNoMoreFrames = true;
                break;
            }

            if (_nowFrame.time < _sharingData._startTime)
                continue;

            if (_1stFrameIndex < 0)
                _1stFrameIndex = _nowFrame.index;

            _mu_nowFrame.lock();

            _nowFrame.index -= _1stFrameIndex;
            _nowFrame.fileName = itosWith0(_nowFrame.index) + ".jpg";

            auto timeEnd = chrono::high_resolution_clock::now();
            chrono::duration<double, nano> elapsed = timeEnd - timeStart;
            _nowFrame.timeInReading = elapsed.count() / 1e9;

            _mu_nowFrame.unlock();

            _mu_isFrontendTracking.lock();
            _isFrontendTracking = true;
            _mu_isFrontendTracking.unlock();

            _mu_isBackendTracking.lock();
            _isBackendTracking = true;
            _mu_isBackendTracking.unlock();

            showFramePoseInfo();

            elapsed = chrono::high_resolution_clock::now() - timeStart;
            if (elapsed.count() > _nowFrame.toNextTime * 1e6)
                cout << "Warning! Time spent in reading frame is larger than 1 / framerate!" << endl;
            else
               _mySleep(((int) (_nowFrame.toNextTime * 1e6 - elapsed.count())))

    #ifdef _SHOW_SLEEP_TIME
            cout << "frame updating time: " << (chrono::high_resolution_clock::now() - timeStart).count() / 1e6 << " (ms)" << endl;
    #endif
        };
    } catch (exception& e) {
        cout << "catch error in updateFrame with: " << e.what() << endl;
    }

    frameMetaFile.close();
}

void logPCM(int frameRate)
{
    ofstream ofp;
    ofp.open(_sharingData._pcmAnzPath);

    ofp << "Frame rate: " << frameRate << endl;
    ofp << "Max PCM offset: " << getPCMRecord().largestOffset << " in " << getPCMRecord().largestOffsetName << endl;;
    ofp << "Average PCM offset: " << getPCMRecord().average() << endl;

    cout << "Frame rate: " << frameRate << endl;
    cout << "Max PCM offset: " << getPCMRecord().largestOffset << " in " << getPCMRecord().largestOffsetName << endl;;
    cout << "Average PCM offset: " << getPCMRecord().average() << endl;

    ofp.close();
}

void afterInit()
{
    addDurationAndNextTime(_sharingData._imuPath, _sharingData._imu2Path, 0);
    addDurationAndNextTime(_sharingData._imagePosePath, _sharingData._imagePose2Path, 1);
    cullFrames();
    _trackingResult.clear();

    ofstream dfcm;
    dfcm.open(_sharingData._dfcmPath, std::ios_base::app);

    _status = SystemStatus_Ini;
    _isNoMoreFrames = false;
    _isBackendTracking = false;
    _isFrontendTracking = false;
    _1stFrameIndex = -1;
    _lastFramePose.isActive = false;
    _isTrackingEnd = false;

    cout << "The pose on backend is GT: " << !_sharingData._isGTPoseBias << endl;
    dfcm << "The pose on backend is GT: " << !_sharingData._isGTPoseBias << endl;
    dfcm.close();

    if (_trackingPoseFP.is_open())
        _trackingPoseFP.close();
    remove(_sharingData._trackingPosePath.c_str());
    _trackingPoseFP.open(_sharingData._trackingPosePath);
}

void afterTracking()
{
    if (_trackingPoseFP.is_open())
        _trackingPoseFP.close();

    logPCM(_sharingData._frameRate);
}

void tracking(int frameRateExp, int imuRate)
{
    vector<char> serials;
    if (init(frameRateExp, imuRate, serials))
        afterInit();
    else
        return;

    cout << "Start tracking asynchronously..." << endl;

    thread threadFrame(updateFrame, serials);
    thread threadIMU(updateIMU, nullptr, -1);
    thread threadBackTrack(backendTracking, false, true, true);
    thread threadFrontTrack(frontendTracking, false);
    thread threadSaveTrackingResult(saveTrackingPoseRoutine);

    threadFrame.join();
    threadIMU.join();
    threadBackTrack.join();
    threadFrontTrack.join();
    _isTrackingEnd = true;
    threadSaveTrackingResult.join();

    afterTracking();
    cout << "Finished tracking asynchronously!" << endl;

    if (!_is2TestTHR2D)
        postProcessing();
}

void trackingSeq(int frameRateExp, int imuRate, bool isWithDealy = true)
{
    vector<char> serials;
    if (init(frameRateExp, imuRate, serials))
        afterInit();
    else
        return;

    cout << "Start tracking synchronously..." << endl;

    ifstream frameMetaFile;
    frameMetaFile.open(_sharingData._imagePose2Path);
    ifstream imuMetaFile;
    imuMetaFile.open(_sharingData._imu2Path);

    double responseTime = 0;
    while (1)
    {
        if (!isWithDealy)
            responseTime = -1e-9;

        readFrameInfo(frameMetaFile, _nowFrame, _sharingData, true);
        if (!_nowFrame.status || _nowFrame.time > _sharingData._endTime)
        {
            cout << "There are no more frames left, exit..." << endl;
            _isNoMoreFrames = true;
            break;
        }

        if (_nowFrame.time < _sharingData._startTime)
            continue;
        else
        {
            if (_1stFrameIndex < 0)
            {
                _1stFrameIndex = _nowFrame.index;
                responseTime = -1e-9;
            }

            _nowFrame.index -= _1stFrameIndex;
            _nowFrame.fileName = itosWith0(_nowFrame.index) + ".jpg";
        }

        if (responseTime >= _nowFrame.time)          // frontend pose checking
        {
            updateIMU(&imuMetaFile, _nowFrame.time);
            frontendTracking(_isFrontendTracking = true);
        }
        else                                        // backend pose estimation
        {
            if (_status >= SystemStatus_0_Send)
            {
                updateIMU(&imuMetaFile, responseTime);
                backendTracking(_isBackendTracking = true, false, true);
            }

            updateIMU(&imuMetaFile, _nowFrame.time);
            backendTracking(_isBackendTracking = true, true, false);

            frontendTracking(_isFrontendTracking = true);
            responseTime = _nowFrame.time + _sharingData._timeBack;
        }

        showFramePoseInfo();
    };

    frameMetaFile.close();
    imuMetaFile.close();

    afterTracking();
    cout << "Finished tracking synchronously!" << endl;

    if (!_is2TestTHR2D)
        postProcessing();
}

void calculateAdjacentFramePoseDiff()
{
    cout << "Start to calculate adjacent frame pose difference..." << endl;

    remove(_sharingData._ajPoseDiffPath.c_str());
    ofstream ofp;
    ofp.open(_sharingData._ajPoseDiffPath);

    vector<char> serials;
    DFCM(30, 200, serials);

    long count = 0;
    double total = 0;
    double min = INFINITY;
    double max = -1;

    ifstream frameMetaFile;
    frameMetaFile.open(_sharingData._imagePose2Path);
    while (1)
    {
        FrameInfo last = _nowFrame;
        readFrameInfo(frameMetaFile, _nowFrame, _sharingData, true);

        if (!_nowFrame.status || _nowFrame.time > _sharingData._endTime)
            break;

        if (last.status)
        {
            double dis = getBoundingbox2DPointsDiff(&_object, _cameraK,  BasicPose(_nowFrame.r, _nowFrame.p), BasicPose(last.r, last.p));
            ofp << last.index << " " << _nowFrame.index << fixed << setprecision(6) << " " << dis << endl;

            total += dis;
            count++;
            max = MAX(max, dis);
            min = MIN(min, dis);
        }
    }
    ofp << endl;
    ofp << "Max error: " << max << endl;
    ofp << "Min error: " << min << endl;
    ofp << "Mean error: " << total / count << endl;

    ofp.close();
    frameMetaFile.close();

    cout << "Finished calculating adjacent frame pose difference..." << endl;
}

void test()
{
#ifdef _TEST_EIGEN
    testEigen();
#endif

#ifdef _TEST_FILE
    testFile(_imu2Path);
#endif

#ifdef _TEST_FRAMES
    testFrames("/Users/sherlock/Downloads/simulation/results-duck/image.txt", "/Users/sherlock/Downloads/simulation/results-duck/frames/");
#endif

#ifdef _TEST_GEOMETRY
    testGeometry();
#endif

#ifdef _TEST_IMU
    testIMU(_sharingData, false);
#endif

#ifdef _TEST_PCM
    FrameInfo rFrame, frame;
    string fileName = _baseSimPath + "assets/images/000000.jpg";

    Mat image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty())
    {
        cout << "Cannot open or find the image!" << endl;
        return;
    }
    else
    {
        image.copyTo(rFrame.frame);
        frame.frame = imread(_baseSimPath + "assets/images/000098.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        testPCM(rFrame.frame, frame.frame);
    }
#endif

#ifdef _TEST_GZPRJ
    testGZProjection();
#endif

#ifdef _TEST_BIASW
    testBiasW();
#endif

#ifdef _TEST_DUCK
    testDuck();
#endif
}

void postAnzPCM(int baseFPS, int jump, bool isGTPose = false)
{
    cout << "Start doing post PCM analysis..." << endl;
    remove(_sharingData._pcmAnzPath.c_str());

    ifstream fp;
    fp.open(_sharingData._trackingErrorPath);
    string line;
    getline(fp, line);
    vector<string> d(getLineSub(line, ' '));
    string fakeFileName = d[0];
    fp.close();
    fp.open(_sharingData._trackingPosePath);

    Vector3d pNow, eNow, pLast, eLast;
    bool is1st = true;
    int passed = 0;

    resetPCM();
    while (1)
    {
        string line;
        getline(fp, line);

        if (line.empty() || line.length() < 10)
        {
            cout << "Out of data..." << endl;
            break;
        }
        else
        {
            if (passed % (jump + 1) == 0)
            {
                vector<string> d(getLineSub(line, ' '));

                if (isGTPose)
                {
                    pNow =  Vector3d(atof(d[7].c_str()), atof(d[8].c_str()), atof(d[9].c_str()));
                    eNow =  Vector3d(atof(d[10].c_str()), atof(d[11].c_str()), atof(d[12].c_str()));
                }
                else
                {
                    pNow =  Vector3d(atof(d[1].c_str()), atof(d[2].c_str()), atof(d[3].c_str()));
                    eNow =  Vector3d(atof(d[4].c_str()), atof(d[5].c_str()), atof(d[6].c_str()));
                }

                if (is1st)
                    is1st = false;
                else
                {
                    FrameInfo fi;
                    fi.fileName = d[0];
                    string framePath = _sharingData._baseFramePath + fakeFileName + ".jpg";
                    fi.frame = imread(framePath, CV_LOAD_IMAGE_GRAYSCALE);
                    if (fi.frame.empty())
                    {
                        cout << "Cannot read frame raw data of " << framePath << endl;
                        break;
                    }

                    IMUPose imu; imu.r = eulers2Matrix(eNow(2), eNow(1), eNow(0)); imu.p = pNow;
                    BasicPose lastPose = BasicPose(eulers2Matrix(eLast(2), eLast(1), eLast(0)), pLast);
                    doPCM(_object, _cameraK, fi, imu, lastPose);
                }

                pLast = pNow; eLast = eNow;
            }

            passed = (passed + 1) % (jump + 1);
        }
    };

    fp.close();

    logPCM(baseFPS / (jump + 1));
}

string _settingFilePath = "./trackingSetting.txt";
void loadSettingFile()
{
    ifstream fp; fp.open(_settingFilePath);
    if (fp.is_open())
    {
        string line;
        getline(fp, line);

        {
            vector<string> d(getLineSub(line, ' '));        // _startTime _duration
            if (!line.empty() && line.length() >= 3 && d.size() >= 2)
            {
                _sharingData._startTime = atof(d[0].c_str());
                _sharingData._duration = atof(d[1].c_str());
                _sharingData._endTime = _sharingData._startTime + _sharingData._duration;
            }
        }

        fp.close();
    }
}

enum OPCode
{
    // opCode, frameRate, imuRate / jump, isAsyn, isGTPoseBias, isByRK4, isBSCM, _isBSCMOnlyV, isFixedExtraDelay, extraDelay, pxe, pem
    OP_Tracking_No_Post_Processing = 0,

    // opCode, frameRate, imuRate / jump, isAsyn, isGTPoseBias, isByRK4, isBSCM, _isBSCMOnlyV, isFixedExtraDelay, extraDelay, pxe, pem
    OP_Tracking,

    // opCode, frameRate, jump, isGTPose
    OP_PCM_Analysis,

    OP_Test
};

int main(int argc, char *argv[])
{
//  opCode, frameRate, imuRate / jump, isAsyn / isGTPose, isGTPoseBias, isByRK4, isBSCM, _isBSCMOnlyV, isFixedExtraDelay, extraDelay, pxe, pem
//    int argvs[] = {1, 120, 200, 1, 0, 0, 1, 0, 0, 30, -1, -1};
    int argvs[] = {1, 52, 200, 1, 0, 0, 1, 0, 0, 30, -1, -1};
    int baseArgc = sizeof(argvs) / sizeof(argvs[0]);

    cout << "Commands: ";
    for(int i = 0; i < argc; i++)
        cout << argv[i] << " ";
    cout << endl;

    if (argc > 1)
    {
        string path(argv[1]);
        _sharingData._resetPathes(path);

        for (int i = 2; i < argc; i++)
            if (baseArgc > i - 2)
                argvs[i - 2] = atoi(argv[i]);
    }

    loadSettingFile();
    cout << "Tracking strat time: " << _sharingData._startTime << endl;
    cout << "Tracking end time: " << _sharingData._endTime << endl;

    _is2TestTHR2D = false;
    switch (argvs[0])
    {
        case OP_Tracking_No_Post_Processing:
            _is2TestTHR2D = true;
        case OP_Tracking:
        {
            _sharingData._isGTPoseBias = argvs[4] != 0;
            _isByRK4 = argvs[5] != 0;
            _sharingData._isBSCM = argvs[6] != 0;
            _sharingData._isBSCMOnlyV = argvs[7] != 0;
            _sharingData._isFixedExtraDelay = argvs[8] != 0;
            _sharingData._extraDelay = (double) argvs[9];

            double pxe = (double) argvs[10];
            double pxm = (double) argvs[11];
    //        cout << "pxe:" << pxe << endl;
    //        cout << "pxm:" << pxe << endl;
            if (pxe >= 0 && pxm >= 0)
                setTHR2D(pxe, pxm);

            try {
                if (argvs[3] != 0)
                    tracking(argvs[1], argvs[2]);
                else
                    trackingSeq(argvs[1], argvs[2]);
            } catch (exception& e) {
                cout << "catch error in tracking with: " << e.what() << endl;
            }
            break;
        }
        case OP_PCM_Analysis:
        {
            vector<char> serials;
            if (init(argvs[1], 200, serials, false))
                postAnzPCM(argvs[1], argvs[2], argvs[3]);
            break;
        }
        case OP_Test:
        {
            _sharingData._isGTPoseBias = argvs[4] != 0;

            addDurationAndNextTime(_sharingData._imuPath, _sharingData._imu2Path, 0);
            addDurationAndNextTime(_sharingData._imagePosePath, _sharingData._imagePose2Path, 1);

            test();
            break;
        }

        default:
            break;
    }

    //    calculateAdjacentFramePoseDiff();

    remove(_sharingData._imagePose2Path.c_str());
    remove(_sharingData._imu2Path.c_str());

    return 100;
}
