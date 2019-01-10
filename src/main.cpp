#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "rtcprgb2gray.h"

int main(int argc, char *argv[])
{
    cv::Mat image = cv::imread(argv[1]);
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    cv::imshow("OpenCV BGR2GRAY", image_gray);
    cv::waitKey();
    cv::Mat image_rgb;
    cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
    cv::Mat image_gray2 = rtcprgb2gray(image_rgb);
    cv::imshow("rtcprgb2gray", image_gray2);
    cv::waitKey();
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(50000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(image_gray, cv::noArray(), keypoints, descriptors);
    std::cout << "extracted " << keypoints.size() << " keypoints" << std::endl;
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1));
    cv::imshow("bgr2gray", output);
    cv::waitKey();
    detector->detectAndCompute(image_gray2, cv::noArray(), keypoints, descriptors);
    std::cout << "extracted " << keypoints.size() << " keypoints" << std::endl;
    cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1));
    cv::imshow("rtcprgb2gray", output);
    cv::waitKey();
    //detector = cv::xfeatures2d::VGG::create();
    //detector->detectAndCompute(image_gray2, cv::noArray(), keypoints, descriptors);
    //cv::drawKeypoints(image, keypoints, output, cv::Scalar::all(-1));
    //cv::imshow("VGG", output);
    //cv::waitKey();
}
