#include "ofApp.h"
#include "ofxCv.h"
#include "Utilities.h"

//--------------------------------------------------------------
void ofApp::setup(){
    threshold = 24;
    
    cameraSize_w = 2448;
    cameraSize_h = 3264;
    cameraSmallSize_w = cameraSize_w/60;
    cameraSmallSize_h = cameraSize_h/60;
    riversImgSize_w = 1920;
    riversImgSize_h = 1080;
    riversSmallSize_w = riversImgSize_w / 1;
    riversSmallSize_h = riversImgSize_h / 1;
    
    
    grayImage.allocate(cameraSmallSize_w, cameraSmallSize_h);
    grayBg.allocate(cameraSmallSize_w, cameraSmallSize_h);
    grayDiff.allocate(cameraSmallSize_w, cameraSmallSize_h);
    edgeImage.allocate(cameraSmallSize_w, cameraSmallSize_h);
    grayRivers.allocate(riversImgSize_w, riversImgSize_h);
    edgeRivers.allocate(riversSmallSize_w, riversSmallSize_h);
    
    //掌画像
//    capImage.loadImage("palm2.jpg");
    capImage.loadImage("palm_center.jpg");
    capImage.setImageType(OF_IMAGE_GRAYSCALE);
    capImage.resize(cameraSmallSize_w, cameraSmallSize_h);
    grayImage.setFromPixels(capImage.getPixels(), cameraSmallSize_w, cameraSmallSize_h);
    
    //背景画像
    capBg.loadImage("background.jpg");
    capBg.setImageType(OF_IMAGE_GRAYSCALE);
    capBg.resize(cameraSmallSize_w, cameraSmallSize_h);
    grayBg.setFromPixels(capBg.getPixels(), cameraSmallSize_w, cameraSmallSize_h);
    
    //河川画像
    riversImage.loadImage("./rivers/rivers6.png");
    riversImage.setImageType(OF_IMAGE_GRAYSCALE);
    riversImage.resize(riversSmallSize_w, riversSmallSize_h);
    grayRivers.setFromPixels(riversImage.getPixels(), riversSmallSize_w, riversSmallSize_h);
    
    //背景との差分
    grayDiff.absDiff(grayBg, grayImage);
    grayDiff.threshold(threshold);
    
    //エッジ
    cvCanny(grayImage.getCvImage(), edgeImage.getCvImage(), 30, 70);
    cvCanny(grayRivers.getCvImage(), edgeRivers.getCvImage(), 30, 70);
    
    //特徴点検出器
    detector1 = cv::FeatureDetector::create("SURF");
    detector2 = cv::FeatureDetector::create("SURF");
    
    //特徴量検出器
//    extractor1 = cv::DescriptorExtractor::create("SURF");
//    extractor2 = cv::DescriptorExtractor::create("SURF");
    
    //マッチング
//    matcher = cv::DescriptorMatcher::create("BruteForce");
    
    //テンプレートマッチング
    //    for(int i=0; i<6; i++){
    //        tmach[i] = *cvCreateImage(cvSize(riversSmallSize_w, riversSmallSize_h), 32, 1);
    //    }
    
    cv::Mat mat1;
    cv::Mat mat2;
    mat1 = edgeRivers.getCvImage();
    mat2 = edgeImage.getCvImage();
    result = edgeRivers.getCvImage();
//    cv::matchTemplate(mat1, mat2, result, CV_TM_SQDIFF_NORMED);
    //    cv::matchTemplate(mat1, mat2, result, CV_TM_CCORR_NORMED);
        cv::matchTemplate(mat1, mat2, result, CV_TM_CCOEFF_NORMED);
//    cv::minMaxLoc(result, &minVal, NULL);
    cv::minMaxLoc(result, NULL, &maxVal);
    ofxCv::toOf(result, resultImage);
//    ofLog(OF_LOG_NOTICE, "minVal: %lf", minVal);
    ofLog(OF_LOG_NOTICE, "maxVal: %lf", maxVal);
}

//--------------------------------------------------------------
void ofApp::update(){
    
    //blobs
    contourFinder.findContours(grayDiff, 20, (cameraSmallSize_w*cameraSmallSize_h), 10, true);
    //    contourFinder.findContours(edgeImage, 20, (cameraSmallSize_w*cameraSmallSize_h)/2, 10, true);
    
    //特徴点
    //    matImg = cv::Mat(grayDiff.getCvImage());
    matImg = cv::Mat(edgeImage.getCvImage());
    matImg2 = cv::Mat(edgeRivers.getCvImage());
    detector1->detect(matImg, keypoints1);
    detector2->detect(matImg2, keypoints2);
    
    //特徴量
//    cv::Mat descriptor1;
//    extractor1->compute(matImg, keypoints1, descriptor1);
//    
//    cv::Mat descriptor2;
//    extractor2->compute(matImg2, keypoints2, descriptor2);
    
    //マッチング
//    matcher->match(descriptor1, descriptor2, dmatch);
}

//--------------------------------------------------------------
void ofApp::draw(){
    //背景との差分
    ofSetHexColor(0xffffff);
    grayDiff.draw(20, 20);
    
    grayRivers.draw(20+cameraSmallSize_w*2, 20);
//    edgeRivers.draw(20+cameraSmallSize_w*2, 20);
    
    ofFill();
    ofSetHexColor(0x333333);
    ofRect(20+cameraSmallSize_w, 20, cameraSmallSize_w, cameraSmallSize_h);
    ofSetHexColor(0xffffff);
    
    //blobs
    //    for (int i = 0; i < contourFinder.nBlobs; i++) {
    //        ofSetColor(255);
    //        contourFinder.blobs[i].draw(20+cameraSmallSize_w*2, 20);
    //        ofPoint cp = contourFinder.blobs[i].centroid;
    //        ofCircle(cp.x + 40, cp.y + 50, 110);
    //    }
    
    //エッジ
    edgeImage.draw(20+cameraSmallSize_w, 20);
    
    //特徴点
    //掌画像
    for (vector<cv::KeyPoint>::iterator itk = keypoints1.begin(); itk != keypoints1.end(); itk++) {
        ofSetColor(31, 63, 255);
        ofCircle(20+cameraSmallSize_w + itk->pt.x, 20 + itk->pt.y, 2);
    }
    //河川画像
    for (vector<cv::KeyPoint>::iterator itk = keypoints2.begin(); itk != keypoints2.end(); itk++) {
//        ofCircle(20+cameraSmallSize_w*2 + itk->pt.x, 20 + itk->pt.y, 2);
    }
    
    //マッチング結果
    //    cv::drawMatches(matImg, keypoints1, matImg2, keypoints2, dmatch, result);
    //    IplImage iplimg(result);
    //    resultImage = &iplimg;
    
    //statement
    ofSetHexColor(0xffffff);
    stringstream reportStr;
    reportStr << "bg subtraction and blob detection" << endl
    << "press ' ' to capture bg" << endl
    << "threshold " << threshold << " (press: +/-)" << endl
    << "num blobs found " << contourFinder.nBlobs << ", fps: " << ofGetFrameRate();
    ofDrawBitmapString(reportStr.str(), 20, 600);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch (key){
        case '+':
            threshold ++;
            if (threshold > 255) threshold = 255;
            break;
        case '-':
            threshold --;
            if (threshold < 0) threshold = 0;
            break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 
    
}
