#include <windows.h>
#include "include/Detector.h"
#include <iostream>
#include <vector>
#include <algorithm> // For std::max_element

// Utility function to convert std::string to std::wstring
std::wstring s2ws(const std::string& s) {
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    std::wstring r(len, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, &r[0], len);
    return r;
}

// Constructor for the Detector class using ONNX Runtime
Detector::Detector(const std::string& modelPath, const std::vector<std::string>& classNames)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-Detector"),
      session(nullptr) {

    this->confidenceThreshold = 0.50f;
    this->nmsThreshold = 0.2f;
    this->classNames = classNames;

    // ONNX Runtime requires a wide-character string for the model path on Windows
    std::wstring modelPathW = s2ws(modelPath);
    session = Ort::Session(env, modelPathW.c_str(), Ort::SessionOptions{nullptr});

    // Get input and output node details from the model
    Ort::AllocatorWithDefaultOptions allocator;

    // Input nodes
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatedStringPtr name_ptr = session.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(name_ptr.get());
    }
    inputNodeNamesPtr.resize(inputNodeNames.size());
    for (size_t i = 0; i < inputNodeNames.size(); i++) {
        inputNodeNamesPtr[i] = inputNodeNames[i].c_str();
    }

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputNodeDims = inputTensorInfo.GetShape();

    // Output nodes
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr name_ptr = session.GetOutputNameAllocated(i, allocator);
        outputNodeNames.push_back(name_ptr.get());
    }
    outputNodeNamesPtr.resize(outputNodeNames.size());
    for (size_t i = 0; i < outputNodeNames.size(); i++) {
        outputNodeNamesPtr[i] = outputNodeNames[i].c_str();
    }
}

// Main detection function using ONNX Runtime
std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    // --- 1. Pre-process the image ---
    const int input_width = static_cast<int>(inputNodeDims[3]);
    const int input_height = static_cast<int>(inputNodeDims[2]);

    // Convert BGR to RGB
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Resize and scale the image
    cv::Mat resized_frame;
    cv::resize(rgb_frame, resized_frame, cv::Size(input_width, input_height));
    
    // Convert to float and scale to [0, 1]
    cv::Mat float_frame;
    resized_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    // Reshape data from HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    cv::Mat channels[3];
    cv::split(float_frame, channels);
    memcpy(input_tensor_values.data(), channels[0].data, input_width * input_height * sizeof(float));
    memcpy(input_tensor_values.data() + input_width * input_height, channels[1].data, input_width * input_height * sizeof(float));
    memcpy(input_tensor_values.data() + 2 * input_width * input_height, channels[2].data, input_width * input_height * sizeof(float));


    // --- 2. Create input tensor and run inference ---
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run the session
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, inputNodeNamesPtr.data(), &input_tensor, 1, outputNodeNamesPtr.data(), 1);

    // --- 3. Post-process the output ---
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_proposals = static_cast<int>(output_shape[2]);
    const int num_classes = static_cast<int>(output_shape[1]) - 4;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = frame.cols / (float)input_width;
    float y_factor = frame.rows / (float)input_height;

    // The output is (1, 4 + num_classes, 8400). We need to transpose it.
    for (int i = 0; i < num_proposals; ++i) {
        // Get the class scores for the current proposal
        float* class_scores = raw_output + (4 * num_proposals) + i;
        
        // Find the class with the highest score
        float max_score = 0.0f;
        int class_id = 0;
        for(int j = 0; j < num_classes; ++j) {
            if (class_scores[j * num_proposals] > max_score) {
                max_score = class_scores[j * num_proposals];
                class_id = j;
            }
        }

        if (max_score > confidenceThreshold) {
            confidences.push_back(max_score);
            class_ids.push_back(class_id);

            // Extract bounding box
            float cx = raw_output[0 * num_proposals + i];
            float cy = raw_output[1 * num_proposals + i];
            float w = raw_output[2 * num_proposals + i];
            float h = raw_output[3 * num_proposals + i];

            int left = static_cast<int>((cx - w / 2) * x_factor);
            int top = static_cast<int>((cy - h / 2) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // --- DEBUG: Print number of potential boxes found before NMS ---
    std::cout << "Found " << confidences.size() << " potential boxes before NMS." << std::endl;
    if (!confidences.empty()) {
        auto max_confidence = *std::max_element(confidences.begin(), confidences.end());
        std::cout << "Max confidence found: " << max_confidence << std::endl;
    }


    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, nms_result);

    std::vector<Detection> final_detections;
    for (int idx : nms_result) {
        Detection result;
        result.box = boxes[idx];
        result.confidence = confidences[idx];
        result.classId = class_ids[idx];
        if (result.classId < this->classNames.size()) {
            result.className = this->classNames[result.classId];
        } else {
            result.className = "Unknown";
        }
        final_detections.push_back(result);
    }

    return final_detections;
}
