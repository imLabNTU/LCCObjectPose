# LCCObjectPose
The official implemenation of "Real-Time Object Pose Tracking System With Low Computational Cost for Mobile Devices" in IEEE Journal of Indoor and Seamless Positioning and Navigation, 2023.

## Setup
Using Xcode: 
1.	Download OpenCV v3.4.14.
2.	Open the project of FrontendTracking.xcodeproj.
3.	Relink the lib of OpenCV to the project.
4.	Compile and get FrontendTracking.exe.

## Run the code
1.	Put `trackingSetting .txt` with FrontendTracking.exe in the same folder (`basePath`).

`trackingSetting.txt` contains one line of the tracking starting time and duration in the format of “start_time (in second) duration (in second)”. For example, “3 30” means the program will ignore the data before 3 seconds, and then keeps tracking for 30 seconds after 3 seconds.

2. Run “FrontendTracking.exe” in the format of:
```
./FrontendTracking.exe $datasetPath 1 $frameRate $imuRate 1 0 0 1 0 0 $extraDelay
```
For example:
```
./FrontendTracking.exe ./ LCCODS/cat-1/ 1 52 200 1 0 0 1 0 0 30
```
The arguments are listed as follows. 
* 1st parameter (datasetPath): the file path of tracking dataset (with LCCODS)
* 2nd parameter: the mode of program, the default is 1 for tracking.
* 3rd parameter (frameRate): the tracking frame rate
* 4th parameter (imuRate): the IMU sample rate
* 5th parameter: if the tracking is asynchronous?
* 6th parameter: if the backend pose (ground truth) is needed to add random noise?
* 7th parameter: if the pose updated with IMU data by RK4?
* 8th parameter: if the BSCM is on?
* 9th parameter: if the BSCM only corrects the system velocity?
* 10th parameter: if extraDelay is fixed-added?
* 11th parameter (extraDelay): the extra delay in networking

## Output Files
You can find two files of tracking pose and error in `basePath`.

1. `tracking_pose.txt`: It contains the tracking pose corresponding with the frame. The format is “frame_name t.x t.y t.z e.x e.y e.z GT.t.x GT.t.y GT.t.z GT.e.x GT.e.y GT.e.z isPassedbyPCM”.
* t means the translation in pose.
* e means the Euler angle in pose.
* GT means the ground truth.

2. `tracking_error.txt`: It contains the tracking pose error corresponding with the frame. The format is “frame_index error_2d error_t error_t.x error_t.y error_t.z error_e error_e.x error_e.y error_e.z isPassedbyPCM”.
* error_2d means the 2d projection error.
* error_t means the translation error in pose.
* error_e means the Euler angle error in pose.

## Reference
If you find this research helpful, please cite it as
```
@article{LCCobjpose,
  author={Lau, Yo-Chung and Tseng, Kuan-Wei and Kao, Peng-Yuan and Hsieh, I-Ju and Tseng, Hsiao-Ching and Hung, Yi-Ping},
  journal={IEEE Journal of Indoor and Seamless Positioning and Navigation}, 
  title={Real-Time Object Pose Tracking System With Low Computational Cost for Mobile Devices}, 
  year={2023},
  volume={1},
  number={},
  pages={211-220},
  doi={10.1109/JISPIN.2023.3340987}}
```
