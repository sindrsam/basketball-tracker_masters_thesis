cmake_minimum_required(VERSION 3.10)
project(BasketballTracker)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package and its components
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio dnn)

# Add the executable with only the necessary source files
add_executable(BasketballTracker 
    src/main.cpp
    src/Detector.cpp
    src/Tracker.cpp
)

# Tell CMake where to find our header files (e.g., "include/Detector.h")
target_include_directories(BasketballTracker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link against the OpenCV libraries that were found
target_link_libraries(BasketballTracker PRIVATE ${OpenCV_LIBRARIES})