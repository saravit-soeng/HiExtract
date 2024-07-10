import sys
sys.path.append('./python')

from hierarchy_extract import HiExtract
from ultralytics import YOLO
import json

# Initial YOLO model and inference
model = YOLO('models/yolov8s.pt')
source = 'assets/images/106903459-1624896118131-gettyimages-1324274760-pi-2189332.jpeg'
result = model.predict(source=source)[0]

# Extract object hierarchy
hi_extract = HiExtract(base_objects=['person'], detection_result=result)
h_result = hi_extract.extract_object_hierarchy()

# Pretty print using json
print(json.dumps(h_result, indent=4))