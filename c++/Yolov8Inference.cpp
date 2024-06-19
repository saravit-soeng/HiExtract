#include "Yolov8Inference.h"

// Function: letterbox for keeping the ratio before resizing
Mat letterbox(const Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

// Function: compileModel for compiling yolo model on openvino runtime
ov::CompiledModel compileModel(string modelPath, ov::Core core, string device) {
    auto compiledModel = core.compile_model(modelPath, device);
    return compiledModel;
}

// Function: preprocess for preprocssing input image
Mat preprocessImage(Mat img, int imgSz, float* scale) {
    Mat letterboxImg = letterbox(img);
    *scale = letterboxImg.size[0] / float(imgSz);
    Mat blob = blobFromImage(letterboxImg, 1.0 / 255.0, Size(imgSz, imgSz), Scalar(), true);

    return blob;
}

// Function: inference for creating creating inference request pre
ov::Tensor createInference(ov::CompiledModel compiledModel, Mat img) {
    ov::InferRequest inferRequest = compiledModel.create_infer_request();

    auto inputPort = compiledModel.input();
    ov::Tensor inputTensor(inputPort.get_element_type(), inputPort.get_shape(), img.ptr(0));
    inferRequest.set_input_tensor(inputTensor);
    inferRequest.infer();

    auto output = inferRequest.get_output_tensor(0);
    return output;
}

// Function: postprocess for post-processing the result
vector<DetectionResult> postprocess(ov::Tensor output, float scale) {
    float* data = output.data<float>();
    auto outputShape = output.get_shape();
    Mat outputBuffer(outputShape[1], outputShape[2], CV_32F, data);
    transpose(outputBuffer, outputBuffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> classIds;
    std::vector<float> classScores;
    std::vector<Rect> boxes;
    vector<float> confidences;
    vector<int> xCenters;
    vector<int> yCenters;
    vector<int> x0List;
    vector<int> y0List;
    vector<int> x1List;
    vector<int> y1List;

    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < outputBuffer.rows; i++) {
        Mat classes_scores = outputBuffer.row(i).colRange(4, 84);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            classScores.push_back(maxClassScore);
            classIds.push_back(class_id.x);
            float cx = outputBuffer.at<float>(i, 0);
            float cy = outputBuffer.at<float>(i, 1);
            float w = outputBuffer.at<float>(i, 2);
            float h = outputBuffer.at<float>(i, 3);

            int left = round((cx - 0.5 * w) * scale);
            int top = round((cy - 0.5 * h) * scale);
            int width = round(w * scale);
            int height = round(h * scale);

            xCenters.push_back(int(cx * scale));
            yCenters.push_back(int(cy * scale));
            x0List.push_back(left);
            y0List.push_back(top);
            x1List.push_back(left + width);
            y1List.push_back(top + height);

            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back(maxClassScore);
        }
    }
    // NMS: Non-max suppression
    std::vector<int> indices;
    NMSBoxes(boxes, classScores, score_threshold, nms_threshold, indices);

    vector<DetectionResult> results;
    for (int i = 0;i < indices.size();i++) {
        int idx = indices[i];
        DetectionResult result{};
        result.classId = classIds[idx];
        result.confidence = confidences[idx];
        result.xCenter = xCenters[idx];
        result.yCenter = yCenters[idx];
        result.x0 = x0List[idx];
        result.y0 = y0List[idx];
        result.x1 = x1List[idx];
        result.y1 = y1List[idx];
        results.push_back(result);
    }

    return results;
}