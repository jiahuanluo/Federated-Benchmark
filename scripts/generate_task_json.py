import os
import json
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd

sets = [('2007', 'train'), ('2007', 'test')]
classes = ["basket", "carton", "chair", "electrombile", "gastank", "sunshade", "table"]
model = "yolo"


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(anno_path, label_path, image_id):
    in_file = open(os.path.join(anno_path, image_id + ".xml"))
    out_file = open(os.path.join(label_path, image_id + ".txt"), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if model == "fasterrcnn":
    for i in range(1, 21):
        dir_path = os.path.join(str(i), "ImageSets", "Main", "train.txt")
        train_size = len(open(dir_path).readlines())
        task_file_path = os.path.join("task_configs", "faster_rcnn_task" + str(i) + ".json")
        task_config = dict()
        task_config["model_name"] = "FasterRCNN"
        task_config["model_config_file"] = "data/task_configs/street_20/faster_rcnn_model.json"
        task_config["log_filename"] = "logs/street_20/FL_street" + str(i) + "_log"
        task_config["data_path"] = "../object_detection/street_20/" + str(i)
        task_config["epoch"] = 5
        task_config["train_size"] = train_size
        task_config["test_size"] = 191

        with open(task_file_path, "w") as f:
            json.dump(task_config, f)


elif model == "yolo":
    for i in range(1, 6):
        label_path = os.path.join(str(i), "labels")
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        for year, image_set in sets:
            anno_path = os.path.join(str(i), "Annotations")
            image_path = os.path.join(str(i), "ImageSets", "Main", image_set + ".txt")
            image_ids = open(image_path).read().strip().split()
            list_file = open('%s/%s.txt' % (str(i), image_set), 'w')
            for image_id in image_ids:
                list_file.write('%s/street_5/%s/JPEGImages/%s.jpg\n' % ("data", str(i), image_id))
                convert_annotation(anno_path, label_path, image_id)
            list_file.close()
        task_file_path = os.path.join("task_configs", "street_5", "yolo_task_" + str(i) + ".json")
        task_config = dict()
        task_config["model_name"] = "Yolo"
        task_config["model_config"] = "data/task_configs/street_5/yolo_model.json"
        task_config["log_filename"] = "logs/street_5/FL_street" + str(i) + "_log"
        task_config["train"] = "data/street_5/" + str(i) + "/train.txt"
        task_config["test"] = "data/street_5/" + str(i) + "/test.txt"
        task_config["names"] = "data/street_5/classes.names"
        task_config["n_cpu"] = 4
        task_config["local_epoch"] = 4
        task_config["batch_size"] = 1
        with open(task_file_path, "w") as f:
            json.dump(task_config, f)
