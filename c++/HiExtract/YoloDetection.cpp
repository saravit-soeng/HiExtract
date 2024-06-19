#include "YoloDetection.h"

// YOLO class names
vector<String> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

int getClassIndexByNameYolo(const std::string& name, const std::vector<std::string>& classes) {
    for (size_t i = 0; i < classes.size(); ++i) {
        if (classes[i] == name) {
            return static_cast<int>(i);
        }
    }
    std::cerr << "Key not found." << std::endl;
    return -1; // Indicating an error
}

vector<Object> detectOhYolo(vector<string>& baseObjects, vector<DetectionResult> results) {
    vector<Object> parentObjects;

    for (const auto& baseObject : baseObjects) {
        int clsIdx = getClassIndexByNameYolo(baseObject, classNames);
        for (int i = 0; i < results.size(); i++) {
            DetectionResult parent = results[i];
            if (parent.classId == clsIdx) { // Determine it as a parent class
                std::vector<Object> children;
                for (int j = 0; j < results.size(); j++) {
                    DetectionResult child = results[j];
                    if (i != j) {
                        if ((parent.x0 < child.xCenter) && (child.xCenter < parent.x1) && (parent.y0 < child.yCenter) && (child.yCenter < parent.y1)) {
                            Object childObject = { classNames[child.classId], child.x0, child.y0, child.x1, child.y1 };
                            children.push_back(childObject);
                        }
                    }
                }

                if (!children.empty()) {
                    Object parentObject = { classNames[parent.classId], parent.x0, parent.y0, parent.x1, parent.y1, children };
                    parentObjects.push_back(parentObject);
                }
            }
        }
    }

    return parentObjects;
}