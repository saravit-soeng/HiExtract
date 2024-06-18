from enum import Enum

class DetectionType(Enum):
    YOLO = 'yolo'
    SSD = 'ssd'
    FastRCNN = 'fast-rcnn'
    FasterRCNN = 'faster-rcnn'