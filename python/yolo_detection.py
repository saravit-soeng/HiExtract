from utils import get_class_index_by_name_yolo, prepare_bounding_boxes_yolo

# function for detecting object hierarchy (YOLO)
def detect_oh_yolo(base_objects, result):
    # Check for different result format of YOLO versions
    try:
        predicted_cls = result.boxes.cls.detach().cpu().numpy()
        xywh = result.boxes.xywh.detach().cpu().numpy()
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        objects_with_boxes = prepare_bounding_boxes_yolo(predicted_cls, xywh, xyxy)
    except:
        predicted_cls = result.xywh[0].detach().cpu().numpy()
        predicted_cls = [r[-1] for r in predicted_cls]
        xywh = result.xywh[0].detach().cpu().numpy()
        xyxy = result.xyxy[0].detach().cpu().numpy()
        objects_with_boxes = prepare_bounding_boxes_yolo(predicted_cls, xywh, xyxy)
        
    classes = result.names
    result_list = []
    for obj in base_objects:
        cls_idx = get_class_index_by_name_yolo(obj, classes)
        if objects_with_boxes:
            for i, parent_obj in enumerate(objects_with_boxes):
                if parent_obj['class_idx'] == cls_idx:
                    children = []
                    for j, child_obj in enumerate(objects_with_boxes):
                        # Prevent check with its own object
                        if i != j:
                            if (parent_obj['xyxy'][0] < child_obj['xywh'][0] < parent_obj['xyxy'][2]) and (parent_obj['xyxy'][1] < child_obj['xywh'][1] < parent_obj['xyxy'][3]):
                                child = {
                                    "class": classes[child_obj['class_idx']],
                                    'x0': int(child_obj['xyxy'][0]),
                                    'y0': int(child_obj['xyxy'][1]),
                                    'x1': int(child_obj['xyxy'][2]),
                                    'y1': int(child_obj['xyxy'][3])
                                }
                                children.append(child)
                    if children:
                        parent = {
                            "base_object":{
                                "class": classes[parent_obj['class_idx']],
                                'x0': int(parent_obj['xyxy'][0]),
                                'y0': int(parent_obj['xyxy'][1]),
                                'x1': int(parent_obj['xyxy'][2]),
                                'y1': int(parent_obj['xyxy'][3]),
                                "child_objects": children
                            }
                        }
                        result_list.append(parent)

    results = {"data": result_list}
    return results