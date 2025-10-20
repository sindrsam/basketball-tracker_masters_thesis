#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // Use ONNX Runtime
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
    Detector(const std::string& modelPath, const std::vector<std::string>& classNames);

    // Main detection function
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // ONNX Runtime members
    Ort::Env env;
    Ort::Session session;

    // Model input/output details
    std::vector<std::string> inputNodeNames;
    std::vector<const char*> inputNodeNamesPtr;
    std::vector<std::string> outputNodeNames;
    std::vector<const char*> outputNodeNamesPtr;
    std::vector<int64_t> inputNodeDims;

    // Other members
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
};

#endif // DETECTOR_H