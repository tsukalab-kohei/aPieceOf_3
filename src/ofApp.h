#pragma once

#include "ofMain.h"
#include "ofxOpenCV.h"
#include "ofxCv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Utilities.h"

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    int cameraSize_w;
    int cameraSize_h;
    int cameraSmallSize_w;
    int cameraSmallSize_h;
    int riversImgSize_w;
    int riversImgSize_h;
    int riversSmallSize_w;
    int riversSmallSize_h;
    
    int threshold;
    
    double minVal;
    double maxVal;
    
    ofxCvColorImage colorImage;
    ofxCvColorImage colorBg;
    
    ofxCvGrayscaleImage grayImage;
    ofxCvGrayscaleImage grayRivers;
    ofxCvGrayscaleImage grayBg;
    ofxCvGrayscaleImage grayDiff;
    ofxCvGrayscaleImage edgeImage;
    ofxCvGrayscaleImage edgeRivers;
    
    ofImage capImage;
    ofImage capBg;
    ofImage riversImage;
    ofImage resultImage;
    
    ofxCvContourFinder contourFinder;
    
    cv::Mat matImg;
    cv::Mat matImg2;
    cv::Mat cut_img;
    cv::Mat result;
    
    cv::Ptr<cv::FeatureDetector> detector1;
    cv::Ptr<cv::FeatureDetector> detector2;
    
    cv::Ptr<cv::DescriptorExtractor> extractor1;
    cv::Ptr<cv::DescriptorExtractor> extractor2;
    
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<cv::DMatch> dmatch;
};
