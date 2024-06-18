# function for getting class index by class name (YOLO)
def get_class_index_by_name_yolo(name, classes):
    try:
        class_indexes = {v: i for i, v in classes.items()}
        return class_indexes[name]
    except:
        print(f"Class name '{name}' does not exist.")

# function for preparing bounding boxes (YOLO)
def prepare_bounding_boxes_yolo(predicted_cls, xywh, xyxy):
    objects_with_boxes = []
    for i, c in enumerate(predicted_cls):
        objects_with_boxes.append({'class_idx': int(c), 'xywh': xywh[i], 'xyxy': xyxy[i]})
    return objects_with_boxes
