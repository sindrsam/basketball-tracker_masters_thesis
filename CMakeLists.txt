cmake_minimum_required(VERSION 3.10)
project(BasketballTracker)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ONNX Runtime Manual Configuration ---
set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/onnxruntime-win-x64-1.17.3)

# Find OpenCV package and its components (dnn is no longer needed)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio dnn)

# Add the executable with only the necessary source files
add_executable(BasketballTracker 
    src/main.cpp
    src/Detector.cpp
    src/MotorController.cpp
)

# Tell CMake where to find header files
target_include_directories(BasketballTracker PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_DIR}/include
)

# Link against the required libraries
target_link_libraries(BasketballTracker PRIVATE 
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)

# Add a post-build command to copy the ONNX Runtime DLL to the output directory
add_custom_command(TARGET BasketballTracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:BasketballTracker>
)