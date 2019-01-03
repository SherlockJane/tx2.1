//
// Author: Yizhou chen
//

#include "MVCamera.h"
#include "MarkerSensor.h"
#include "Timer.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "videosaver.h"
#include <stdio.h>
#include <thread>

//#define _DBG1FRAME_
#define TX2
#ifdef TX2

#include "serial.h"
#endif // TX2
using namespace std;
using namespace cv;
Size dist_size = Size(640, 480);
int X_bias, Y_bias, pix_x, pix_y;
int status;
#ifdef TX2
// vector<int> CalculateXYPixel(Rect location);
int messnum = 1;
bool transmit_message = 0;
Serial sel;
unsigned char data_send_buf[8];
////--------------------------------------
unsigned char Add_CRC(unsigned char InputBytes[], unsigned char data_lenth) {
  unsigned char byte_crc = 0;
  for (unsigned char i = 0; i < data_lenth; i++) {
    byte_crc += InputBytes[i];
  }
  return byte_crc;
}

void Data_disintegrate(unsigned int Data, unsigned char *LData,
                       unsigned char *HData) {
  *LData = Data & 0XFF;          // 0xFF = 1111 1111
  *HData = (Data & 0xFF00) >> 8; // 0xFF00 = 1111 1111 0000 0000
}
//
void Data_Code(unsigned int x_Data, unsigned int y_Data) {
  int length = 8;
  data_send_buf[0] = 0xFF;
  data_send_buf[1] = length;
  data_send_buf[2] = 0x02;
  //
  Data_disintegrate(x_Data, &data_send_buf[3], &data_send_buf[4]);
  Data_disintegrate(y_Data, &data_send_buf[5], &data_send_buf[6]);
  //
  data_send_buf[length - 1] = Add_CRC(data_send_buf, length - 1);
}
////----------------------

#endif // TX2

string num2str(double i)

{
  stringstream ss;
  ss << i;
  return ss.str();
}
void sel_send() {
  while (1) {
#ifdef TX2
    // transmit msg
    if (transmit_message == 1) {
      if (status == 1) {

        char buff[8];
        messnum++;
        Data_Code(X_bias, Y_bias);
        cout << "x_cord= " << X_bias << "    y_cord=  " << Y_bias << endl;
        for (int i = 0; i < 8; i++)
          buff[i] = data_send_buf[i];
        sel.writeData(buff, 8);
          cout << "     X_bias:  " << X_bias << "   Y_bias:    " << Y_bias<< endl;
      } else {
        char buff[8];
        messnum++;
        Data_Code(850, 850);
        cout << "status=== " << status << endl;
        for (int i = 0; i < 8; i++)
          buff[i] = data_send_buf[i];
        sel.writeData(buff, 8);
      }
    }

#else
    // cout << "vs_test" << endl;

#endif // TX2
    this_thread::sleep_for(chrono::milliseconds(1));
  }
}
int frame_process() {
  Mat srcImg, bgrImg;
  bool isSuccess = 0;
  float X = 0, Y = 0, Z = 0;
  int led_type = 0, false_idx = 0, frameCnt = 0;
  MarkSensor markSensor;
  long timeStamp[2];
  HaarD haarDetector;
  bool ishaar = 1, issave = 0;
  VideoSaver saver;
  /// sdk
  bool auto_wb;
  double exp_time = 10000;
  bool auto_exp = false;
  bool large_resolution = false;

  /**record video**/
  int cnt = 0;
  shared_ptr<cv::VideoWriter> pWriter;

  MVCamera::Init();
  MVCamera::Play();
  MVCamera::SetExposureTime(false, 1000);
  MVCamera::SetLargeResolution(true);

  /// Load cascade
  String cascade_name = "../params/zgdcascade_1.xml";
  if (!haarDetector.detector.load(cascade_name)) {
    printf("--(!)Error loading face cascade\n");
  };
  timeStamp[1] = getTickCount();
  while (true) {
    MVCamera::GetFrame(srcImg);

    frameCnt++;
    if (srcImg.empty()) {
      printf("Image empty !\n");
      continue;
    }
    /// calc fps
    timeStamp[0] = getTickCount();
    float timeFly = (timeStamp[0] - timeStamp[1]) / getTickFrequency();
    float fps = 1 / timeFly;
    timeStamp[1] = getTickCount();
    cout << "-----FPS------" << fps << endl;
    /// detect and track
    resize(srcImg, bgrImg, dist_size);
    // bgrImg=srcImg.clone();
    if (!ishaar)
      isSuccess = markSensor.ProcessFrameLEDXYZ(bgrImg, X, Y, Z, led_type,
                                                X_bias, Y_bias);
    else
      isSuccess =
          haarDetector.Detect_track(bgrImg, X, Y, Z, led_type, X_bias, Y_bias);
    if (isSuccess) {
      status = 0;
      cout << "detected target" << endl;
      X_bias = pix_x - 320;
      Y_bias = pix_y - 240;

    } else {
      status = 1;
    }

    /// write demo
    // VideoWriter writer;	//¶šÒå¹ØÓÚÊÓÆµÐŽ³öÀàVideoWriterµÄÊµÀý
    // string videoPathCut =
    // "F://vs_ws//cpp_tutorial//RM_zju//demo_save//demo.avi";
    // writer.open(videoPathCut, CV_FOURCC('D', 'I', 'V', 'X'), rate,
    // videoSize); writer << MarkSensor::img_show;
    /// show img
    if (MarkerParams::ifShow) {
      if (MarkSensor::img_show.empty()) {
        break;
      }

      string frameCnt_s = num2str(frameCnt);
      putText(MarkSensor::img_show, frameCnt_s, cv::Point(600, 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
      string fpsString("FPS:");
      string fps_string = num2str(fps);
      fpsString += fps_string;
      putText(MarkSensor::img_show, fpsString, cv::Point(5, 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
      if (issave)
        putText(MarkSensor::img_show, "recording", cv::Point(300, 450),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 200, 255));
      imshow("all markers", MarkSensor::img_show);

      char key = waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
        // writer.release();
      }
      if (key == 'z')
        system("pause");
      if (key == 's') {
        false_idx++;
        string saveName_src =
            "../dbg_imgs/" + num2str(false_idx) + "falsesrc.jpg";
        imwrite(saveName_src, srcImg);
        string saveName_post =
            "../dbg_imgs/" + num2str(false_idx) + "false_post.jpg";
        imwrite(saveName_post, MarkSensor::img_show);
      }
      if (key == 'a') {
        auto_exp = !auto_exp;
        MVCamera::SetExposureTime(auto_exp, exp_time);
        double read_exp_time = MVCamera::GetExposureTime();
        printf("read_exp_time: %lf\n", read_exp_time);
      }

      if (key == 'p') {
        exp_time += 1000;
        MVCamera::SetExposureTime(auto_exp, exp_time);
        double read_exp_time = MVCamera::GetExposureTime();
        printf("read_exp_time: %lf\n", read_exp_time);
      }

      if (key == 'm') {
        if (exp_time - 1000 > 0)
          exp_time -= 1000;
        MVCamera::SetExposureTime(auto_exp, exp_time);
        double read_exp_time = MVCamera::GetExposureTime();
        printf("read_exp_time: %lf\n", read_exp_time);
      }

      if (key == 'b') {
        large_resolution = !large_resolution;
        MVCamera::SetLargeResolution(large_resolution);
      }
      if (key == 'c') {
        ishaar = !ishaar;
      }
      if (key == 'r') {
        issave = !issave;
      }
    }
    if (issave) {
      saver.write(MarkSensor::img_show);
    }
  }
  MVCamera::Stop();
  MVCamera::Uninit();
  return 0;
}
int main()
{
#ifdef TX2

	//set param of sel

	sel.setPara();
#endif 
	thread th1(&frame_process);
	thread th2(&sel_send);
	th2.join();
	th1.join();

}
