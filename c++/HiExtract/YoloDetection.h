#pragma once
#include<opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#ifndef YoloDetection_H
#define YoloDetection_H

extern vector<String> classNames;

struct Object {
    string className;
    int x0, y0, x1, y1; // Coordinates of the object's bounding box
    vector<Object> children; // Child objects
};

// Struct for storing detection results
struct DetectionResult {
    int classId;
    float confidence;
    int xCenter;
    int yCenter;
    int x0;
    int y0;
    int x1;
    int y1;
};

//function for getting class index by class name 
int getClassIndexByNameYolo(const std::string& name, const std::vector<std::string>& classes);

// function for extracting object hierarchy based on YOLO
vector<Object> detectOhYolo(vector<string>& baseObjects, vector<DetectionResult> results);

#endif