#pragma once

#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>
#include "Detector.h"

class MotorController {
public:
    MotorController();
    void update(const std::vector<Detection>& detections, const cv::Size& frameSize);

private:
    // --- Tracking & Prediction State ---
    std::optional<cv::Rect> trackedPlayerBox;
    std::optional<cv::Rect> prevTrackedPlayerBox; // Store previous position for velocity calculation
    double last_detection_time;                  // Timestamp of the last successful detection
    int frames_since_last_seen;                  // For safety stop
    int hand_signal_consecutive_frames;        // Counter for pass signal stability
    bool is_aligned;                            // Flag for when the tracker is on target

    void findBestPlayer(const std::vector<Detection>& detections);

    // --- From PIDController.cpp ---
    // PID state variables (previous error, integral term, etc.) for pan
    double pan_pid_error_last;
    double pan_pid_integral;

    void calculateMotorCommands(const cv::Rect& targetBox, const cv::Size& frameSize, double dt);

    // --- Hardware Communication ---
    void sendCommandsToHardware(double pan_command);
};