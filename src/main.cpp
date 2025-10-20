#include "include/Detector.h"
#include "include/MotorController.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

int main() {
    // --- Model and Video Paths ---
    std::string custom_model_path = "model/best.onnx";
    std::vector<std::string> class_names = {"Hand-Signal", "Player"};

    // --- Initialization ---
    Detector detector(custom_model_path, class_names);
    MotorController motorController;

    // Open the default webcam
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    const std::string window_name = "Basketball Tracker";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // --- Main Loop ---
    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cout << "End of video stream." << std::endl;
            break;
        }

        // 1. Detect all objects
        std::vector<Detection> all_detections = detector.detect(frame);

        // 2. Update the motor controller
        motorController.update(all_detections, frame.size());

        // 3. Draw bounding boxes for all detections (for debugging)
        for (const auto& detection : all_detections) {
            cv::Scalar color;
            if (detection.className == "Player") {
                color = cv::Scalar(255, 0, 0); // Blue for Player
            } else if (detection.className == "Hand-Signal") {
                color = cv::Scalar(0, 255, 0); // Green for Hand-Signal
            }

            cv::rectangle(frame, detection.box, color, 2);

            // Add a label with class name and confidence
            std::string label = detection.className + ": " + cv::format("%.2f", detection.confidence);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(frame, label, cv::Point(detection.box.x, detection.box.y - labelSize.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // 4. Display Frame
        cv::imshow(window_name, frame);

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // --- Cleanup ---
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
