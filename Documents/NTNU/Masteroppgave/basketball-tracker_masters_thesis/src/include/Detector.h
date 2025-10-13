#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

// A simple struct to hold the result of a detection
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    std::string className;
};

class Detector {
public:
    // Constructor loads the model
    Detector(const std::string& modelPath, const std::string& configPath, const std::string& classesPath);

    // Main detection function
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;

    // Helper function to get output layer names
    std::vector<std::string> getOutputLayerNames();
};

#endif // DETECTOR_H