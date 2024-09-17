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

###################################################################################################################

address = '/home/fariborz_taherkhani/train/trainSwin.json'

file = open(address, 'r')

data = json.load(file)
print(data['categories'])
# stop_sign=0
# bboxes=[]
# for an in data['annotations']:
#     #print(an['image_id'])
#     if an['image_id'] == 1075:
#         bboxes.append(an['bbox'])
#         print(an['image_id'], an['category_id'], an['id'], an['bbox'])
#
# for im in data['images']:
#     print(im['id'])
#     if im['id'] == 1075:
#         im_add = '/home/fariborz/PycharmProjects/datasetSplit/train/images/'+im['file_name']
#         img = cv2.imread(im_add)
#         for bbx in bboxes:
#             p1 = (bbx[0], bbx[1]);
#             p2 = (bbx[0] + bbx[2], bbx[1] + bbx[3])
#             cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
#
#         plt.imshow(img)
#         plt.show()
#
# print(len(data['annotations']))
#

images_ = '/home/fariborz_taherkhani/train/images'

register_coco_instances("experiment", {}, address, images_)

sample_metadata = MetadataCatalog.get("experiment")
dataset_dicts = DatasetCatalog.get("experiment")

import time
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import torch
import  numpy as np
from segment_anything import sam_model_registry
#import supervision as sv
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH='sam_vit_h_4b8939.pth'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
print(DEVICE)
import cv2
from shapely.geometry import Polygon
from segment_anything import SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(sam)

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou
counter=0
id_list = ['Envelope', 'Magazine', 'Newspaper', 'Polybag', 'Debris', 'Flats', 'Box']
AREA=[]
for index_ in range(len(id_list)):
    AREA.append([])

for d in (dataset_dicts):
            image_bgr= cv2.imread(d["file_name"])
            # plt.imshow(image_bgr)
            # plt.show()
            # visualizer = Visualizer(image_bgr[:, :, ::-1], metadata=sample_metadata, scale=0.5)
            # vis = visualizer.draw_dataset_dict(d)
            # # cv2.imshow(vis.get_image()[:, :, ::-1])
            # plt.imshow(vis.get_image())
            # plt.show()
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            #result = mask_generator.generate(image_rgb)
            ans = []
            for object in d['annotations']:
                mask_gr = np.zeros((480, 640), dtype=np.uint8)
                mask_shape = (480, 640)
                for segment in object['segmentation']:
                    polygon = segment
                    # Convert the polygon to a binary mask
                    polygon = np.array(polygon).reshape((-1, 2))
                    polygon = Polygon(polygon)
                    rr, cc = np.array(polygon.exterior.coords.xy).astype(np.int32)
                    cv2.fillPoly(mask_gr, [np.array([rr, cc]).T], color=1)
                mask_dic = {}
                mask_dic['mask'] = mask_gr
                # plt.imshow(mask_gr)
                # plt.show()
                mask_dic['category_id'] = object['category_id']
                area = sum(sum(mask_gr))
                AREA[object['category_id']].append(area)
                ans.append(mask_dic)
                folder_name = object['category_id']
                folder_address = 'SAM_DATASET/' + id_list[folder_name]
                if not os.path.exists(folder_address):
                    os.makedirs(folder_address)
                if area > 600:
                    img_address = folder_address+'/'+ str(area) + '_' + str(counter)+'.jpg'
                else:
                    folder_address = 'SAM_DATASET/' + 'things'
                    img_address = folder_address + '/' + str(area) + '_' + id_list[folder_name] + '_' + str(counter) + '.jpg'
                counter += 1
                rgb_image_mask = np.zeros((mask_gr.shape[0], mask_gr.shape[1], 3), dtype=np.uint8)
                rgb_image_mask[:, :, 0] = mask_gr
                rgb_image_mask[:, :, 1] = mask_gr
                rgb_image_mask[:, :, 2] = mask_gr
                msk_img = rgb_image_mask * image_rgb
                bbx = object['bbox']
                p1 = (bbx[0], bbx[1]);
                p2 = (bbx[0] + bbx[2], bbx[1] + bbx[3])
                cv2.rectangle(msk_img, p1, p2, (0, 255, 0), 2)

                cv2.imwrite(img_address, msk_img)


save_it = np.asarray(AREA)
np.save('area', save_it)

Finish=0


            # for item in result:
            #     mask = item['segmentation']
            #     mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            #     rgb_image_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            #     rgb_image_mask[:, :, 0] = mask
            #     rgb_image_mask[:, :, 1] = mask
            #     rgb_image_mask[:, :, 2] = mask
            #     Flag = False
            #     for mask_condid in ans:
            #         iou = compute_iou(mask, mask_condid['mask'])
            #         if iou >= 0.7:
            #             Flag = True
            #             msk_img = rgb_image_mask*image_rgb
            #             folder_name = mask_condid['category_id']
            #             folder_address = 'SAM_DATASET/' + id_list[folder_name]
            #             if not os.path.exists(folder_address):
            #                 os.makedirs(folder_address)
            #             img_address = folder_address+'/'+str(counter)+'.jpg'
            #             counter += 1
            #             cv2.imwrite(img_address, msk_img)
            #             break
            #     # if Flag == False:
            #     #         folder_address = 'SAM_DATASET/' + 'garbage/'
            #     #         img_address = folder_address + '/' + str(counter) + '.jpg'
            #     #         counter += 1
            #     #         msk_img = rgb_image_mask * image_rgb
            #     #         cv2.imwrite(img_address, msk_img)
            #     print('stop')
