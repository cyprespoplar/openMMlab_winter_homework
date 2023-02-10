import os
import json
import cv2

def get_json(path):
    f = open(path+"/via_region_data.json")
    dic = json.load(f)
    f.close()
    ann = {}
    images = []
    categories = []
    annotations = {}
    os.chdir(path)
    bbox_id = 0
    for i, item in enumerate(dic.keys()):
        file_name = dic[item]["filename"]
        region = dic[item]["regions"]
        h, w = cv2.imread(file_name).shape[:2]
        images.append(dict(file_name=file_name, height=h, width=w, id=i))
        for i in region:
            x, y = region[i]["shape_attributes"]["all_points_x"], region[i]["shape_attributes"]["all_points_y"]
            bbox = [min(x), min(y), max(x) - min(x), max(y) - min(y)]
            area = bbox[2] * bbox[3]
            bbox_id += 1
            print(bbox)
            annotations.append(
                {"segmentation": [[]], "area": area, "iscrowd": 0, "image_id":int(i), "category_id": 1, "id": bbox_id,
                 "bbox": bbox})
    categories = [
        {'supercategory': 'balloon', 'id': 1, 'name': 'balloon'},
        {'supercategory': 'background', 'id': 0, 'name': 'background'}]
    ann["annotations"] = annotations
    ann["images"] = images
    ann["categories"] = categories
    os.chdir("../annotations")
    final = open(path+".json", "w")
    json.dump(ann, final)
    final.close()
    os.chdir("../")
os.chdir("data/balloon")
os.mkdir("annotations")
path_train = "train"
path_val = "val"
get_json(path_train)
get_json(path_val)


