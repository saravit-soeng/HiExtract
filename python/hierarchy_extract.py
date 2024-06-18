from yolo_detection import detect_oh_yolo
from detype_enums import DetectionType

class HiExtract():
    """
    This class is used to handle the extraction of object hierarchy from a given set of base objects based on a detection result.

    Attributes:
        base_objects (list): A list of base objects from which the extraction will be performed.
        detection_result (object): An object representing the detection result that will be used to determine which objects to extract.
        dtype (DetectionType): The type of detection algorithm used to generate the detection result. Defaults to DetectionType.YOLO.
    """
    def __init__(self, base_objects, detection_result, dtype=DetectionType.YOLO):
        self.base_objects = base_objects
        self.detection_result = detection_result
        self.dtype = dtype
    
    # Define function for detecting object hierarchy
    def detect_object_hierarchy(self):
        results = {}
        if self.dtype == DetectionType.YOLO:
            results = detect_oh_yolo(self.base_objects, self.detection_result)
        return results
        