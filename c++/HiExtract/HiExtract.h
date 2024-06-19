#pragma once

#include "YoloDetection.h"

using namespace std;
using namespace cv;

#ifndef HiExtract_H
#define HiExtract_H

enum class DetectionType {
    YOLO = 0,
    SSD,
    FastRCNN,
    FasterRCNN
};

vector<Object> extractObjectHierarchy(vector<string>& baseObjects, vector<DetectionResult> results, DetectionType dtype);

#endif