#include "include/MotorController.h"
#include <iostream>
#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>

// --- PID & Prediction Constants - These will need tuning! ---
const double PAN_KP = 0.5;
const double PAN_KI = 0.01;
const double PAN_KD = 0.1;
const double LEAD_TIME_S = 0.5; // (t) - Predict 0.5 seconds into the future
const int MAX_FRAMES_WITHOUT_DETECTION = 15;
const double MIN_COMMAND = -255.0;
const double MAX_COMMAND = 255.0;


MotorController::MotorController() 
    : trackedPlayerBox(std::nullopt),
      prevTrackedPlayerBox(std::nullopt),
      last_detection_time(0),
      frames_since_last_seen(0),
      pan_pid_error_last(0),
      pan_pid_integral(0) {
    std::cout << "MotorController initialized." << std::endl;
}

void MotorController::update(const std::vector<Detection>& detections, const cv::Size& frameSize) {
    // --- Time Calculation for Velocity ---
    double current_time = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    double dt = (last_detection_time > 0) ? (current_time - last_detection_time) : 0.0;

    // 1. Find the best player to track
    findBestPlayer(detections);

    // 2. If a player is being tracked, calculate motor commands
    if (trackedPlayerBox.has_value()) {
        frames_since_last_seen = 0; // Reset safety counter
        calculateMotorCommands(trackedPlayerBox.value(), frameSize, dt);
        last_detection_time = current_time; // Update time of last successful detection
    } else {
        // --- Safety Stop Logic ---
        frames_since_last_seen++;
        if (frames_since_last_seen > MAX_FRAMES_WITHOUT_DETECTION && frames_since_last_seen < MAX_FRAMES_WITHOUT_DETECTION + 5) {
             // Only send stop command once to avoid flooding the serial port
            std::cout << "No player detected for " << MAX_FRAMES_WITHOUT_DETECTION << " frames. Stopping motors." << std::endl;
            sendCommandsToHardware(0.0); 
        }
        prevTrackedPlayerBox = std::nullopt; // Invalidate previous position
    }

    // 3. Check for a hand signal to trigger a pass
    for (const auto& detection : detections) {
        if (detection.className == "Hand-Signal") {
            std::cout << "--- PASS COMMAND DETECTED ---" << std::endl;
            // sendCommandsToHardware(0, 0); // Example: stop tracking and pass
            break; // Assume one signal is enough
        }
    }
}

void MotorController::findBestPlayer(const std::vector<Detection>& detections) {
    double maxArea = 0;
    std::optional<cv::Rect> bestCandidate;

    for (const auto& detection : detections) {
        if (detection.className == "Player") {
            double area = detection.box.area();
            if (area > maxArea) {
                maxArea = area;
                bestCandidate = detection.box;
            }
        }
    }
    // Update current and previous boxes for velocity calculation
    prevTrackedPlayerBox = trackedPlayerBox;
    trackedPlayerBox = bestCandidate;
}

void MotorController::calculateMotorCommands(const cv::Rect& targetBox, const cv::Size& frameSize, double dt) {
    // Get center of the frame
    double frameCenterX = frameSize.width / 2.0;

    // Get center of the current target box
    double targetCenterX = targetBox.x + targetBox.width / 2.0;

    // --- Predictive Tracking ---
    double predictedTargetCenterX = targetCenterX;
    if (prevTrackedPlayerBox.has_value() && dt > 0) {
        double prevTargetCenterX = prevTrackedPlayerBox->x + prevTrackedPlayerBox->width / 2.0;
        double velocityX = (targetCenterX - prevTargetCenterX) / dt; // pixels per second
        predictedTargetCenterX = targetCenterX + (velocityX * LEAD_TIME_S);
    }

    // Calculate error based on the *predicted* position
    double pan_error = predictedTargetCenterX - frameCenterX;

    // --- PID Calculation for Pan ---
    pan_pid_integral += pan_error;
    double pan_derivative = (dt > 0) ? ((pan_error - pan_pid_error_last) / dt) : 0;
    double pan_command = (PAN_KP * pan_error) + (PAN_KI * pan_pid_integral) + (PAN_KD * pan_derivative);
    pan_pid_error_last = pan_error;

    // Clamp the command to a safe range for the hardware
    double clamped_pan_command = std::clamp(pan_command, MIN_COMMAND, MAX_COMMAND);

    sendCommandsToHardware(clamped_pan_command);
}

void MotorController::sendCommandsToHardware(double pan_command) {
    // Placeholder for actual hardware communication (e.g., Serial, UDP)
    // For now, we just print the commands to the console.
    std::cout << "Motor Commands -> Pan: " << std::fixed << std::setprecision(2) << pan_command << std::endl;
}
