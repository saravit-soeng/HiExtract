#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "HiExtract/YoloDetection.h"

using namespace std;
using namespace cv;
using namespace dnn;

#ifndef YOLOv8Inference_H
#define YOLOv8Inference_H

// Function: letterbox for keeping the ratio before resizing
Mat letterbox(const Mat& source);

// Function: compileModel for compiling yolo model on openvino runtime
ov::CompiledModel compileModel(string modelPath, ov::Core core, string device);

// Function: preprocess for preprocssing input image
Mat preprocessImage(Mat img, int imgSz, float* scale);

// Function: inference for creating creating inference request pre
ov::Tensor createInference(ov::CompiledModel compiledModel, Mat img);

// Function: postprocess for post-processing the result
vector<DetectionResult> postprocess(ov::Tensor output, float scale);

#endif // !YOLOv8Inference_H
