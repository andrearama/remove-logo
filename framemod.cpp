#include <iostream>
#include <math.h>
#include<framemod.hpp>

#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/photo.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

void enlargeRect(Rect& rect, int nPixel = 5){
    // Padding the rectangle by nPixel 
    rect.x -= nPixel; 
    rect.y -= nPixel; 
    rect.height +=nPixel*2; 
    rect.width += nPixel*2;
}

bool compareRectangles(Rect rect1, Rect rect2, int distMax = 200){
    // Compute the difference of centers.
    float dx, dy;
    dx = (rect1.x + rect1.width/2) - (rect2.x + rect2.width/2);
    dy = (rect1.y + rect1.height/2) - (rect2.y + rect2.height/2);
    float distRects = std::sqrt(dx*dx + dy*dy);
    return (distRects<distMax);
}

void mergeRectangles(Rect& rect1, const Rect& rect2){
    if(rect2.x+rect2.width>rect1.x+rect1.width) {rect1.width=(int)rect2.x+(int)rect2.width-min(rect1.x,rect2.x);}else{ rect1.width=(int)rect1.x+(int)rect1.width-min(rect1.x,rect2.x);}
    if(rect2.y+rect2.height>rect1.y+rect1.height){rect1.height=(int)rect2.y+(int)rect2.height-min(rect1.y,rect2.y);} else{rect1.height=(int)rect1.y+(int)rect1.height-min(rect1.y,rect2.y);}
    rect1.x = min(rect1.x,rect2.x);
    rect1.y = min(rect1.y,rect2.y); 
};

vector<Rect> getMergedRectangles(vector<Rect> rectangles ){
    // Merge (greedily) iteratively the rectangles until convergence
    int l = 1;
    bool isMerge = false;
    while(l !=rectangles.size() && rectangles.size()>1) {
        l = rectangles.size();
        Rect currentRect = rectangles[rectangles.size()-1];
        rectangles.pop_back();
     
        for (Rect& e : rectangles){
            isMerge = compareRectangles(e, currentRect);
            if(isMerge){mergeRectangles(e,currentRect);break;}
        }
     
        if(!isMerge){rectangles.push_back(currentRect);}
    }
    return rectangles;
};


void createFrameMod(Mat &rgb, vector<Rect>& ROIs, vector<vector<Point>> boundaries){

    Mat small;
    cvtColor(rgb, small, cv::COLOR_BGR2GRAY);

    Mat maskWhite;
    adaptiveThreshold(small,maskWhite,255.0,0, 0,25, -2.5);
    int edSize = 3;
    Mat element = getStructuringElement( MORPH_RECT,
                Size( 2*edSize + 1, 2*edSize+1 ),
                Point( edSize, edSize ) );  
    dilate(maskWhite,maskWhite,element );
    erode(maskWhite,maskWhite,Mat() );
    dilate(maskWhite,maskWhite,Mat() );
    
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);

    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);

    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    // get the rectangles
    std::vector<Rect> rectangles;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        Rect rect = boundingRect(contours[idx]);
        
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), FILLED);

        RotatedRect rrect = minAreaRect(contours[idx]);
        double r = (double)countNonZero(maskROI) / (rrect.size.width * rrect.size.height);

        bool inBoundaries = false;
        for (auto bound : boundaries){
            if(rect.x> bound[0].x && rect.y>bound[0].y && 
               rect.y+rect.height<bound[1].y && rect.x+rect.width<bound[1].x ){
                inBoundaries = true;
                break;
            }
        }
        // assume at least 25% of the area is filled if it contains text, and not too small
        if (r > 0.25 && (rrect.size.height > 8 && rrect.size.width > 8) && inBoundaries){

            rectangles.push_back(rect);
        }
    }
    rectangles = getMergedRectangles(rectangles);
    for (auto rect : rectangles){
        enlargeRect(rect);
        Mat rez = rgb(rect);
        //Remove text
        Mat cropMaskW = maskWhite(rect);
        inpaint(rez, cropMaskW, rez, 10, INPAINT_TELEA );
        inpaint(rez, cropMaskW, rez, 11, INPAINT_NS );
        //rectangle(rgb, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0, 255, 0), 2);
    }
    ROIs = rectangles;
};


void frameModifier::createBackSub(){
        BackSubInit = true;
        pBackSub = createBackgroundSubtractorMOG2(500,16,true);
}

    
frameModifier::frameModifier(){BackSubInit = false;}

void frameModifier::getRefImage(Mat frame, vector<vector<Point>> boundaries){
    refFrameOr = frame.clone();
    createFrameMod(frame, ROIs, boundaries);
    refFrameMod = frame;
}

void frameModifier::saveImage(){
    imwrite("refFrameOr.jpg",refFrameOr);
    imwrite("refFrameMod.jpg",refFrameMod);
}
    

Mat frameModifier::computeDiffImage(Mat& frame, int frameNumber, bool useMOG2 = false){
    int edSize = 2;
    Mat element = getStructuringElement( MORPH_RECT,
                Size( 2*edSize + 1, 2*edSize+1 ),
                Point( edSize, edSize ) );    
    Mat fgMask;
    if (! useMOG2){
        absdiff(refFrameOr,frame,fgMask);
        cvtColor(fgMask, fgMask, cv::COLOR_BGR2GRAY);
        threshold(fgMask, fgMask, 25, 255.0,cv::THRESH_BINARY);
        erode(fgMask,fgMask,Mat() );dilate(fgMask,fgMask,Mat());
    }
    //for MOG to work, need to set refImage as initial frame
    else{
        if(! BackSubInit){createBackSub();}
        pBackSub->apply(frame, fgMask);
        erode(fgMask,fgMask,element );dilate(fgMask,fgMask,element);
        dilate(fgMask,fgMask,element);erode(fgMask,fgMask,element );
    }
    return fgMask;
}

void frameModifier::modifyImage(Mat& frame, int frameNumber = 0){   

    Mat fgMask = computeDiffImage(frame, frameNumber);
    Mat fgMask1 = fgMask.clone();
    Mat refFrameMod1 = refFrameMod.clone();
    Mat frame1 = frame.clone();

    for (const Rect& rect: ROIs){

        Mat maskCrop(fgMask1, rect);
        Mat refCrop(refFrameMod1, rect);   
        Mat refMasked = frame1(rect);

        refCrop.copyTo(refMasked, 255- maskCrop);

        Mat insetImage(frame, rect);
        refMasked.copyTo(insetImage);
        
    }
}    
    