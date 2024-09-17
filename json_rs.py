import json
import os.path
from collections import  defaultdict
# address = os.path.join(os.getcwd(),'test/testSwin.json')
address = '/home/fariborz_taherkhani/Updated_Sam_Model/output/inference/coco_instances_results.json'
file = open(address, 'r')

data = json.load(file)
Hello=0
# print(data['images'][10])
# for item in data['annotations']:
#     print(len(item))
#     wait=0
# for item in data['annotations']:
#     print(item['segmentation'])
#
# print(data['images'])
# counter=defaultdict(lambda:0)
# # print(data['annotations'])
# print('hello')
# # print(data.keys())
# # print(data['categories'])
# # for ann in data['annotations']:
# #     if ann['category_id']==5:
# #         print(ann['id'])
# #     # print(ann['categories'])
# #     # counter[ann['category_id']]+=1
# # print(counter)
# # #print(data['annotations'])
# # p=0
# for ann in data['annotations']:
#     ann['iscrowd'] = 0
#     ann['area'] = ann['bbox'][2]*ann['bbox'][3]
#
# # Save the updated annotations to a new file
# address = os.path.join(os.getcwd(),'test/updated_testPlayment.json')
# with open(address, 'w') as f:
#     json.dump(data, f)
