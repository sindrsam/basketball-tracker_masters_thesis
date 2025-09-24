# Real-Time Player Tracking and Control of a Robotic Basketball System

This repository contains the source code and documentation for my Master's thesis at NTNU. The project aims to give a "The Gun 6000" basketball rebounding machine the ability to dynamically track a player on the court and respond to hand gestures.

## Project Goal

The primary goal is to transform a static, location-based training machine into a dynamic, interactive training partner by implementing a real-time computer vision and closed-loop control system.

## Core Components

*   **Hardware:** A modified "The Gun 6000" machine, an Arduino microcontroller for low-level control, and a laptop with a webcam for vision processing.
*   **Vision Software (C++):** Uses OpenCV and a YOLO model to detect the player's position and hand gestures.
*   **Control Software (C++ & Arduino):** A PID controller implemented in C++ sends commands to the Arduino, which directly controls the machine's panning motor and encoder.

## How to Build and Run

*(You will fill this out later)*

1.  **Prerequisites:**
    *   CMake 3.15+
    *   OpenCV 4.x
    *   A C++17 compiler (e.g., GCC, MSVC)
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/sindrsam/master-thesis-basketball-tracker.git
    cd master-thesis-basketball-tracker
    ```
3.  **Build the project:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
4.  **Run the application:**
    ```bash
    ./basketball_tracker
    ```
