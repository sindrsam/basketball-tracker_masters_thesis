#include "include/Detector.h"
#include <fstream>
#include <iostream>

Detector::Detector(const std::string& modelPath, const std::string& configPath, const std::string& classesPath) {
    // Initialize thresholds
    confidenceThreshold = 0.5;
    nmsThreshold = 0.4;

    // Load the neural network
    net = cv::dnn::readNetFromDarknet(configPath, modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::ifstream ifs(classesPath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open classes file: " + classesPath);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        classNames.push_back(line);
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    // Use swapRB=true, which was the original, working setting
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputLayerNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(out.at<float>(i, 0) * frame.cols);
                int centerY = static_cast<int>(out.at<float>(i, 1) * frame.rows);
                int width = static_cast<int>(out.at<float>(i, 2) * frame.cols);
                int height = static_cast<int>(out.at<float>(i, 3) * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection detection;
        detection.box = boxes[idx];
        detection.classId = classIds[idx];
        detection.confidence = confidences[idx];
        if (detection.classId < classNames.size()) {
            detection.className = classNames[detection.classId];
        } else {
            detection.className = "Unknown";
        }
        detections.push_back(detection);
    }

    return detections;
}

std::vector<std::string> Detector::getOutputLayerNames() {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}