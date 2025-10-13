#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/Detector.h"
#include "include/Tracker.h"

int main() {
    try {
        // Initialize detector for person detection
        Detector detector("model/yolov3-tiny.weights", "model/yolov3-tiny.cfg", "model/coco.names");
        Tracker tracker;

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }

        cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break;
            }

            // Person detection
            std::vector<Detection> detections = detector.detect(frame);

            // Update tracker with new detections
            tracker.update(detections);

            // Get the tracked person
            auto tracked_person = tracker.getTrackedPlayerBox();

            if (tracked_person) {
                // Draw bounding box for the tracked person
                cv::rectangle(frame, *tracked_person, cv::Scalar(255, 0, 0), 2);
            }

            cv::imshow("Frame", frame);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}