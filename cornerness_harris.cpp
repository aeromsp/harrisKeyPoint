#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
//    cv::waitKey(0);

    vector<cv::KeyPoint> keypnts;
    double maxOverlap = 0.0; 
    for (size_t i = 0; i < dst_norm.rows; i++)
    {
        for (size_t j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i, j);
            if (response > minResponse){ //first condition > threshold
                cv::KeyPoint tmpKeyPoint;
                tmpKeyPoint.pt = cv::Point2f(j, i);
                tmpKeyPoint.size = 2 * apertureSize;
                tmpKeyPoint.response = response;

                bool overlap_b = false;
                for (auto kpit = keypnts.begin(); kpit != keypnts.end(); ++kpit)
                {
                    double kptOverlap = cv::KeyPoint::overlap(tmpKeyPoint, *kpit);
                    if (kptOverlap > maxOverlap)
                    {
                        overlap_b = true;
                        if (tmpKeyPoint.response > (*kpit).response)
                        {                      
                            *kpit = tmpKeyPoint; 
                            break;             
                        }
                    }
                }
                if (!overlap_b)
                {                                     
                    keypnts.push_back(tmpKeyPoint); 
                }
            }
        } 
    }     

    windowName = "Harris Corner Detector";
    cv::namedWindow("Harris Corner Detector", 5);
    cv::Mat harCorDet = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypnts, harCorDet, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Harris Corner Detector", harCorDet);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}
