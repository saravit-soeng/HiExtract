#include "YoloDetection.h"
#include "HiExtract.h"

vector<Object> extractObjectHierarchy(vector<string>& baseObjects, vector<DetectionResult> results, DetectionType dtype)
{	
	vector<Object> parentObjects;
	if (dtype == DetectionType::YOLO)
		parentObjects = detectOhYolo(baseObjects, results);
		
	return parentObjects;
}

