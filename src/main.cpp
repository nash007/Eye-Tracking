#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>


#include<stdio.h>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

using namespace cv;

/** Constants **/

int lpx1=0; int lpy1=0;

int rx=06;int ry=06;
int lpx,lpy,rpx,rpy;
int countert,counterl;
double tl,tr,bl,br,ll,lr,rr,rl;
double tal,tar,lal,lar,rar,ral,bal,bar;
int arr[500][500][2];
bool stage1 = 1,stage2;


/** Function Headers */
void detectAndDisplay( cv::Mat frame );
void MyFilledCircle( Mat img, Point center );
/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;					// CascadeClassifier -> class to detect objects in a video stream.
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);									// RNG -> Random Number Generator
cv::Mat debugImage;									// Mat -> Basic Image comtainer
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

/**
 * @function main
 */
int main( int argc, const char** argv ) {

	srand(time(NULL)); int ra;

/*	char circ_window[] = "Moving dot";

	Mat circ_image = Mat::zeros( 400, 400, CV_8UC3 );
	MyFilledCircle( circ_image, Point( 100, 100) );
	imshow( circ_window, circ_image );
	cv::setWindowProperty( circ_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	moveWindow( circ_window, 900, 200 );*/
	//sleep(8);

	CvCapture* capture;
	cv::Mat frame;
	
	// Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

	cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
	cv::moveWindow(main_window_name, 400, 100);
	cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
	cv::moveWindow(face_window_name, 10, 100);
	cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
	cv::moveWindow("Right Eye", 10, 600);
	cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
	cv::moveWindow("Left Eye", 10, 800);
	cv::namedWindow("aa",CV_WINDOW_NORMAL);
	cv::moveWindow("aa", 10, 800);
	cv::namedWindow("aaa",CV_WINDOW_NORMAL);
	cv::moveWindow("aaa", 10, 800);

	createCornerKernels();
	ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
			43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

	// Read the video stream
	capture = cvCaptureFromCAM( -1 );
	if( capture ) {
		while( true ) {
	
	char circ_window[] = "Moving dot";

	Mat circ_image = Mat::zeros( 414, 414, CV_8UC3 );
	//ra = rand()%4;
	//if (ra==1) rx+=1; else if(ra==2) rx-=1; else if(ra==3) ry+=1; else ry-=1; rx+=1; if(rx==500) rx=0;

	if(stage1 && !stage2)
		if(rx>=6 && rx <=400 && ry==6)
		{
			rx+= 10;
			tl+= lpy;
			tr+= rpy;
			countert ++;
		}
		else if(rx>=400 && ry<400)
		{
			ry+=10;
			rl+= lpx;
			rr+= rpx;
			counterl ++;
		}
		else if(ry>=400 && rx > 6)
		{
			rx-=10;
			bl+= lpy;
			br+= rpy;
		}
		else if(rx<=6 && ry>20)
		{
			ry-=10;
			ll+= lpx;
			lr+= rpx;
		}
		else if(rx <= 6 && ry <= 20 && ry > 6)
		{
			stage1 = 0;
			stage2 = 1;
		}
	if(!stage1 && stage2)
	{
		tal = tl / countert;
		tar = tr / countert;
		bar = br / countert;
		bal = bl / countert;
		lal = ll / (counterl-1);
		lar = lr / (counterl-1);
		ral = rl / counterl;
		rar = rr / counterl;
		std::cout<<tal<<" : "<<tar<<" : "<<lal<<" : "<<lar<<std::endl;
		std::cout<<ral<<" : "<<rar<<" : "<<bal<<" : "<<bar<<std::endl;
		stage2 = 0;
		rx=200;ry=200;
	}

	if(!stage1 && !stage2)
	{
		if( //lpx >= lal && 
			lpx <= ral &&
			//rpx >= lar &&
			rpx <= rar &&
			//lpy >= tal &&
			lpy <= bal &&
			//rpy >= tar &&
			rpy <= bar )
			std::cout<<"INSIDE\n";
		else
			std::cout<<"OUTSIDE\n";
	}

	/*	if(rx<200 ) {rx++; px=100; py=100;}
		else if(rx<400) {rx++; px=200; py=300;}
		else if(rx < 600) {rx++; px=400; py =200;}
		else rx=0;*/
	arr[rx][ry][0]=lpx1; arr[rx][ry][0]=lpy1;

	//int px,py;
	MyFilledCircle( circ_image, Point( rx, ry) );
	setWindowProperty( circ_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	imshow( circ_window, circ_image );
	moveWindow( circ_window, 00, 00 );




	frame = cvQueryFrame( capture );
	// Reducing the resolution of the image to increase speed
	cv::Mat smallFrame;
	cv::resize(frame,smallFrame,cv::Size(round(1*frame.cols), round(1*frame.rows)),1,1,cv::INTER_LANCZOS4);

	// mirror it
	cv::flip(smallFrame, smallFrame, 1);
	smallFrame.copyTo(debugImage);

	// Apply the classifier to the frame
	if( !smallFrame.empty() ) {
		detectAndDisplay( smallFrame );
	}
	else {
		printf(" --(!) No captured frame -- Break!");
		break;
	}

	imshow(main_window_name,debugImage);

	int c = cv::waitKey(10);
	if( (char)c == 'c' ) { break; }
	if( (char)c == 'f' ) {
		imwrite("frame.png",smallFrame);
	}

		}
	}

	releaseCornerKernels();

	return 0;
}

void MyFilledCircle( Mat img, Point center )
{
	int thickness = -1;
	int lineType = 8;

	circle( img,
			center,
			6, //radius, probably depends on width
			Scalar( 0, 0, 255 ),
			thickness,
			lineType );
}


void findEyes(cv::Mat frame_gray, cv::Rect face) {

	clock_t start,end;
	double time_taken;
	start = clock();

	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;

	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
	}
	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth/100.0);
	int eye_region_height = face.width * (kEyePercentHeight/100.0);
	int eye_region_top = face.height * (kEyePercentTop/100.0);
	cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
			eye_region_top,eye_region_width,eye_region_height);
	cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
			eye_region_top,eye_region_width,eye_region_height);

	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
	cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");

	int lpx2 = leftPupil.x;
	int lpy2 = leftPupil.y;
	int thresh = 6;
	if (abs(lpx2-lpx1) < thresh)
	{
		//std::cout<<"lp "<<leftPupil<<" ";
		//std::cout<<"rp "<<rightPupil<<std::endl;
	}

	lpx1 = lpx2; lpy1 = lpy2;

	lpx = leftPupil.x;
	lpy = leftPupil.y;
	rpx = rightPupil.x;
	rpy = rightPupil.y;

	/*char circ_window[] = "Moving dot";
	  Mat circ_image = Mat::zeros( 400, 400, CV_8UC3 );
	  MyFilledCircle( circ_image, Point( 100, 100) );
	  imshow( circ_window, circ_image );
	  setWindowProperty( circ_window, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//moveWindow( circ_window, 900, 200 );

	MyFilledCircle( circ_image, Point( lpx1, lpy1) );
	imshow( circ_window, circ_image );*/


	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);
	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
	rectangle(debugFace,leftRightCornerRegion,200);
	rectangle(debugFace,leftLeftCornerRegion,200);
	rectangle(debugFace,rightLeftCornerRegion,200);
	rectangle(debugFace,rightRightCornerRegion,200);
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	// draw eye centers
	circle(debugFace, rightPupil, 3, 1234);
	circle(debugFace, leftPupil, 3, 1234);

	//-- Find Eye Corners
	if (kEnableEyeCorner) {
		cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
		leftRightCorner.x += leftRightCornerRegion.x;
		leftRightCorner.y += leftRightCornerRegion.y;
		cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
		leftLeftCorner.x += leftLeftCornerRegion.x;
		leftLeftCorner.y += leftLeftCornerRegion.y;
		cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
		rightLeftCorner.x += rightLeftCornerRegion.x;
		rightLeftCorner.y += rightLeftCornerRegion.y;
		cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
		rightRightCorner.x += rightRightCornerRegion.x;
		rightRightCorner.y += rightRightCornerRegion.y;
		circle(faceROI, leftRightCorner, 3, 200);
		circle(faceROI, leftLeftCorner, 3, 200);
		circle(faceROI, rightLeftCorner, 3, 200);
		circle(faceROI, rightRightCorner, 3, 200);
	}

	imshow(face_window_name, faceROI);

	end = clock();
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	//std::cout<<"Time taken by the find eyes function : "<<time_taken<<std::endl;
	//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
	//  cv::Mat destinationROI = debugImage( roi );
	//  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
	clock_t start,end;
	double time_taken;
	start = clock();

	cv::Mat input;
	cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

	cvtColor(frame, input, CV_BGR2YCrCb);

	for (int y = 0; y < input.rows; ++y) {
		const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
		//    uchar *Or = output.ptr<uchar>(y);
		cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
		for (int x = 0; x < input.cols; ++x) {
			cv::Vec3b ycrcb = Mr[x];
			//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
			if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
				Or[x] = cv::Vec3b(0,0,0);
			}
		}
	}

	end = clock();
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	//std::cout<<"Time taken by the find skin function : "<<time_taken<<std::endl;

	return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
	clock_t start,end;
	double time_taken;
	start = clock();

	std::vector<cv::Rect> faces;
	//cv::Mat frame_gray;

	std::vector<cv::Mat> rgbChannels(3);
	cv::split(frame, rgbChannels);
	cv::Mat frame_gray = rgbChannels[2];

	//cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//equalizeHist( frame_gray, frame_gray );
	//cv::pow(frame_gray, CV_64F, frame_gray);

	//-- Detect faces
	//Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(80, 80) );

	//  findSkin(debugImage);

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(debugImage, faces[i], 1234);
	}
	//-- Show what you got
	if (faces.size() > 0) {
		findEyes(frame_gray, faces[0]);
	}

	end = clock();
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	//std::cout<<"Time taken by the detect and display function : "<<time_taken<<std::endl;
}
