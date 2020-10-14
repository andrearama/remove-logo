#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/photo.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <framemod.hpp>

using namespace cv;
using namespace std;

//hardcoded boundaries for logo removal
vector<vector<Point>> boundaries {{Point(830,400),Point(1185,535)},
                                            {Point(120,450),Point(518,585)},
                                            {Point(120,5),Point(1238,100)} };
const int frame_width = 1366;
const int frame_height = 768;

int main()
{
    string filename = "video2.mp4";
    VideoCapture capture(filename);
    Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    namedWindow( "window", 1);

    VideoWriter video("outcpp.avi",VideoWriter::fourcc('M','J','P','G'),15, Size(frame_width,frame_height)); 


    Mat refImg = imread("refImage.jpg");
    frameModifier frameMod;
    frameMod.getRefImage(refImg, boundaries);
    frameMod.saveImage();

    for(int i = 0; ;i++)
    {
        capture >> frame;
        cv::resize(frame, frame, cv::Size(frame_width,frame_height), 0, 0);
        if(frame.empty())
            break;

        // remove logos
        frameMod.modifyImage(frame, i); 

        video.write(frame);
        imshow("w", frame);
        waitKey(20);
    }
    capture.release();
    video.release();

}