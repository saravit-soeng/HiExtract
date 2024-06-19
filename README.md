# HiExtract
HiExtract: A simple and efficient technique for extracting object hierarchies based on object detection

![figure-Page-5](https://github.com/saravit-soeng/HiExtract/assets/19525030/84f953c1-98b6-47dc-8bfb-6c098b5ab7ee)


### Abstract
An object hierarchy refers to the structured relationship between objects, where parent objects have one or more child objects. This hierarchical structure is useful in various applications, such as detecting motorcycle riders without helmets or identifying individuals carrying illegal items in restricted areas. However, extracting object hierarchies from images or videos is challenging without advanced techniques like machine learning or deep learning. In this research, we propose a simple and efficient method for extracting object hierarchies based on object detection results. We implemented this method in a standalone package compatible with both Python and C++ programming languages. The package processes object detection results to produce object hierarchies by applying overlapping criteria on bounding boxes to determine parent-child relationships. Our results demonstrate that the proposed approach can accurately extract object hierarchies from images and videos, offering a practical tool for enhancing object detection capabilities. The developed source code for this approach is released at [https://github.com/saravit-soeng/HiExtract](https://github.com/saravit-soeng/HiExtract)

#

### Installation
```
git clone https://github.com/saravit-soeng/HiExtract
cd HiExtract
```

### Usage Example
##### Python
```
from hierarchy_extract import HiExtract
from ultralytics import YOLO
import json

# Initial YOLO model and inference
model = YOLO('models/yolov8s.pt')
source = 'assets/images/106903459-1624896118131-gettyimages-1324274760-pi-2189332.jpeg'
result = model.predict(source=source)[0]

# Extract object hierarchy
hi_extract = HiExtract(base_objects=['person'], detection_result=result)
h_result = hi_extract.detect_object_hierarchy()

# Pretty print using json
print(json.dumps(h_result, indent=4))
```
#### C++
```
#include "Yolov8Inference.h"
#include "HiExtract/HiExtract.h"

int main() {
    try{
        // Path to Yolov5 Model in IR format
        // Model folders is under the same directory of main.cpp file
        string modelPath = "yolov8s_openvino_model/yolov8s.xml";

        // Define input size
        int imgsz = 640;

        // Define devices (CPU | GPU) for inference
        string device = "CPU";
        
        string imagePath = "E:/Code-Workspace/Other/HiExtract/assets/images/106903459-1624896118131-gettyimages-1324274760-pi-2189332.jpeg";
        Mat frame = imread(imagePath);

        ov::Core core;
        cout << "Start compiling model" << endl;
        auto model = compileModel(modelPath, core, device);

        float scale = 0.0;
        auto resizedImage = preprocessImage(frame, imgsz, &scale);
        auto output = createInference(model, resizedImage);
        auto detectionResults = postprocess(output, scale);

        vector<string> base_objects = { "person" };
        vector<Object> parentObjects = extractObjectHierarchy(base_objects, detectionResults, DetectionType::YOLO);

        for (const auto& object : parentObjects) {
            cout << "- Parent Object:" << object.className << endl;
            vector<Object> children = object.children;
            for (const auto& child : children) {
                cout << "==> Child name: " << child.className << endl;
            }
        }
    }
    catch (Exception ex) {
        cerr << ex.msg;
    }

    return 0;
}
```

The above sample code for c++ is inferenced on OpenVino runtime using YOLOv8s.

#### Sample result
![figure-Page-10](https://github.com/saravit-soeng/HiExtract/assets/19525030/6d2527bd-f262-4be2-9b0e-65d1cb32d287)
