
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;
void enlargeRect(Rect& rect, int nPixel); 
bool compareRectangles(Rect rect1, Rect rect2, int distMax);
void mergeRectangles(Rect& rect1, const Rect& rect2);
vector<Rect> getMergedRectangles(vector<Rect> rectangles );
void createFrameMod(Mat &rgb, vector<Rect>& ROIs, vector<vector<Point>> boundaries);

class frameModifier{
    Mat refFrameOr;
    Mat refFrameMod;
    bool BackSubInit;
    Ptr<BackgroundSubtractor> pBackSub;
    std::vector<Rect> ROIs;

    void compareImage(Mat& frame);
    void createBackSub();

    public:
    frameModifier();
    void getRefImage(Mat frame, vector<vector<Point>> boundaries);
    void saveImage();
    Mat computeDiffImage(Mat& frame, int frameNumber, bool useMOG2);
    void modifyImage(Mat& frame, int frameNumber);
};