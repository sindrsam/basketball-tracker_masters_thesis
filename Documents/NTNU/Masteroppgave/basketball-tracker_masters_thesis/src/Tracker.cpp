#include "include/Tracker.h"
#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>

Tracker::Tracker() : trackedPlayerBox(std::nullopt) {}

void Tracker::update(const std::vector<Detection>& detections) {
    double maxArea = 0;
    std::optional<cv::Rect> bestCandidate;

    for (const auto& detection : detections) {
        if (detection.className == "person") {
            double area = detection.box.area();
            if (area > maxArea) {
                maxArea = area;
                bestCandidate = detection.box;
            }
        }
    }

    trackedPlayerBox = bestCandidate;
}

std::optional<cv::Rect> Tracker::getTrackedPlayerBox() const {
    return trackedPlayerBox;
}