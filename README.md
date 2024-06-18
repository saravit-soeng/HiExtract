# HiExtract
HiExtract: A simple and efficient technique for extracting object hierarchies based on object detection

![figure-1](https://github.com/saravit-soeng/HiExtract/assets/19525030/dd317910-6714-4270-8c74-0f4b0aa642de)

### Abstract
An object hierarchy refers to the structured relationship between objects, where parent objects have one or more child objects. This hierarchical structure is useful in various applications, such as detecting motorcycle riders without helmets or identifying individuals carrying illegal items in restricted areas. However, extracting object hierarchies from images or videos is challenging without advanced techniques like machine learning or deep learning. In this research, we propose a simple and efficient method for extracting object hierarchies based on object detection results. We implemented this method in a standalone package compatible with both Python and C++ programming languages. The package processes object detection results to produce object hierarchies by applying overlapping criteria on bounding boxes to determine parent-child relationships. Our results demonstrate that the proposed approach can accurately extract object hierarchies from images and videos, offering a practical tool for enhancing object detection capabilities. The developed source code for this approach is released at [https://github.com/saravit-soeng/HiExtract](https://github.com/saravit-soeng/HiExtract)

### Install
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
// To be added later...
```

#### Sample result
![figure-2](https://github.com/saravit-soeng/HiExtract/assets/19525030/d7291403-744b-41b2-97a3-f3809accbc44)
