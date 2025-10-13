#ifndef TRACKER_H
#define TRACKER_H

#include "Detector.h"       // We need the Detection struct
#include <opencv2/core.hpp> // For cv::Rect
#include <optional>
#include <vector>           // For std::vector

class Tracker {
public:
    Tracker();

    // Updates the tracker with a new set of detections
    void update(const std::vector<Detection>& detections);

    // Returns the bounding box of the tracked player, if one exists
    std::optional<cv::Rect> getTrackedPlayerBox() const;

private:
    std::optional<cv::Rect> trackedPlayerBox;
};

#endif // TRACKER_H