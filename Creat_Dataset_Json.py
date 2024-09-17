import cv2
import matplotlib.pyplot as plt
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os
import  numpy as np
from shapely.geometry import Polygon
###################################################################################################################
import time
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import torch
import  numpy as np
import cv2
from shapely.geometry import Polygon

address = '/home/fariborz_taherkhani/test/testSwinC.json'

file = open(address, 'r')
data = json.load(file)
extracted_data = {}
for field in data:
    print(field)
    if field != 'annotations':
        extracted_data[field] = data[field]

anotation_list=[]
extracted_data['categories'].append({'id': 8, 'keypoints': ['center', 'bestPickPoint', 'corner1', 'corner2', 'corner3', 'corner4'], 'name': 'Things', 'skeleton': [], 'supercategory': 'bundles'})

for item in data['annotations']:
    print(item['category_id'])
    anotation={}
    mask_gr = np.zeros((480, 640), dtype=np.uint8)
    mask_shape = (480, 640)
    for segment in item['segmentation']:
        polygon = segment
        # Convert the polygon to a binary mask
        polygon = np.array(polygon).reshape((-1, 2))
        polygon = Polygon(polygon)
        rr, cc = np.array(polygon.exterior.coords.xy).astype(np.int32)
        cv2.fillPoly(mask_gr, [np.array([rr, cc]).T], color=1)
    area = sum(sum(mask_gr))
    if area > 600:
       for field_ in item:
           anotation[field_]=item[field_]
    else:
        for field_ in item:
            if field_ != 'category_id':
               anotation[field_] = item[field_]
            else:
                anotation[field_] = 8
    anotation_list.append(anotation)


extracted_data['annotations'] = anotation_list

with open('/home/fariborz_taherkhani/test/test_updated.json', 'w') as destination_file:
    json.dump(extracted_data, destination_file)




images_ = '/home/fariborz_taherkhani/test/images'
address = '/home/fariborz_taherkhani/test/test_updated.json'

register_coco_instances("experiment", {}, address, images_)

sample_metadata = MetadataCatalog.get("experiment")
dataset_dicts = DatasetCatalog.get("experiment")

for d in (dataset_dicts):
            image_bgr= cv2.imread(d["file_name"])
            plt.imshow(image_bgr)
            plt.show()
            visualizer = Visualizer(image_bgr[:, :, ::-1], metadata=sample_metadata, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow(vis.get_image()[:, :, ::-1])
            plt.imshow(vis.get_image())
            plt.show()

