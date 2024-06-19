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

        // Visualize detection result
        for (int i = 0;i < detectionResults.size(); i++) {
            // cout << "==> class: " << classNames[detectionResults[i].classId] << ", conf:" << detectionResults[i].confidence << endl;
            int x0 = detectionResults[i].x0;
            int y0 = detectionResults[i].y0;
            int x1 = detectionResults[i].x1;
            int y1 = detectionResults[i].y1;
            int xCenter = detectionResults[i].xCenter;
            int yCenter = detectionResults[i].yCenter;
            string name = classNames[detectionResults[i].classId];
            float conf = detectionResults[i].confidence;

            Scalar f_color = Scalar(255, 0, 0);

            rectangle(frame, Point(x0, y0), cv::Point(x1, y1), f_color, 2);

            int font = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            stringstream conf_str;
            conf_str << conf;
            string output = name + " " + conf_str.str();
            Size size = getTextSize(output, font, fontScale, thickness, nullptr);

            rectangle(frame, Point(x0, y0 - 20), cv::Point(x0 + size.width, y0), f_color, cv::FILLED);
            putText(frame, output, Point(x0, y0 - 5), font, fontScale, cv::Scalar(255, 255, 255), thickness);
        }

        // Display image of detected objects
        namedWindow("Detection", WINDOW_NORMAL);
        imshow("Detection", frame);
        waitKey();
        destroyAllWindows();
    }
    catch (Exception ex) {
        cerr << ex.msg;
    }

    return 0;
}