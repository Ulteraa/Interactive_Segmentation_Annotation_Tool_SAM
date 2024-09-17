import cv2
import matplotlib.pyplot as plt
import torch, torchvision
print(torch.cuda.is_available())
# print(torch.__version__, torch.cuda.is_available())
# assert torch.__version__.startswith("1.9")  # please manually install torch 1.9 if Colab changes its default version

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

import matplotlib.pyplot as plt
# import some common detectron2 utilities
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
json_folders = 'anotated_json_files/json'
image_folder = 'anotated_json_files/image'
for filename in os.listdir(json_folders):
    address=os.path.join(json_folders,filename)
    image_name = filename.replace('.json', '.jpg')
    images_ = os.path.join(image_folder, image_name)
    register_coco_instances("experiment", {}, address, image_folder)

    sample_metadata = MetadataCatalog.get("experiment")
    dataset_dicts = DatasetCatalog.get("experiment")
    # print(len(dataset_dicts))
    import random
    #thing_classes=['Envelope', 'Magazine', 'Newspaper', 'Polybag', 'Debris', 'Flats', 'Box']


    print(len(dataset_dicts))
    id_=0
    for d in (dataset_dicts):
                print(len(d['annotations']))
                img = cv2.imread(d["file_name"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # plt.imshow(img)
                # plt.show()
                visualizer = Visualizer(img[:, :, ::-1], scale=0.5, instance_mode="bbox")
                vis = visualizer.draw_dataset_dict(d)
                name_=str(id_)+'.png'
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
                ax[0].imshow(img)
                ax[0].set_title("Original Image")
                ax[0].axis('off')

                ax[1].imshow(vis.get_image())
                ax[1].set_title("Output Image")
                ax[1].axis('off')

                plt.show()
                id_+=1
                DatasetCatalog.remove("experiment")
                MetadataCatalog.remove("experiment")
                # address_i=os.path.join(save_add, name_)
                # cv2.imwrite(address_i, vis.get_image())
                # cv2.imshow(vis.get_image()[:, :, ::-1])
                # plt.imshow(vis.get_image())
                # plt.show()