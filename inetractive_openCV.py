# import cv2
# import numpy as np
#
# drawing = False  # True if mouse is pressed
# ix, iy = -1, -1
#
# # Mouse callback function
# def draw_rectangle(event, x, y, flags, param):
#     global ix, iy, drawing
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             img_temp = img.copy()
#             cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
#             cv2.imshow('Interactive Drawing', img_temp)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
#         cv2.imshow('Interactive Drawing', img)
#
# img = np.zeros((512, 512, 3), dtype=np.uint8)
#
# cv2.namedWindow('Interactive Drawing')
# cv2.setMouseCallback('Interactive Drawing', draw_rectangle)
#
# while True:
#     cv2.imshow('Interactive Drawing', img)
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:  # Press 'Esc' to exit
#         break
#
# cv2.destroyAllWindows()
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib
from matplotlib.patches import Rectangle
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
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import torch
import  cv2
import time
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import torch
import  numpy as np
from segment_anything import sam_model_registry
#import supervision as sv

class InteractiveRectangleDrawer:
    def __init__(self, ax, name):
        self.ax = ax
        self.rect = None
        self.x_start = None
        self.y_start = None
        self.x_end = None
        self.y_end = None
        self.rect_params = None
        self.rectangles = []
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.coco_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []}
        self.coco_data["categories"].append({"id": 1, "name": "package"})
        self.name = name
        self.object_id = 0
        self.image_id = os.path.splitext(self.name)[0]
        self.shape = original_image.shape
        imd_dic = {'file_name': self.name, 'id': self.image_id, 'width': int(self.shape[0]), 'height': int(self.shape[1])}
        self.coco_data['images'].append(imd_dic)
    def on_press(self, event):
        if event.button == 1:  # Left click to start drawing
            self.x_start = event.xdata
            self.y_start = event.ydata
            self.rect = plt.Rectangle((self.x_start, self.y_start), 0, 0, linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(self.rect)
            self.rectangles.append(self.rect)
            self.rect_params = {
                'x': self.x_start,
                'y': self.y_start,
                'width': 0,
                'height': 0
            }
            #self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if self.rect is not None and event.button == 1:  # Left click and rectangle started
            self.x_end = event.xdata
            self.y_end = event.ydata
            width = self.x_end - self.x_start
            height = self.y_end - self.y_start
            self.rect.set_width(width)
            self.rect.set_height(height)
            self.rect_params = {
                'x': self.x_start,
                'y': self.y_start,
                'width': width,
                'height': height
            }
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        x = self.rect_params['x'];  y = self.rect_params['y']; width = self.rect_params['width'];  height = self.rect_params['height']

        global  rgb_image_mask, original_image

        if self.rect is not None and width > 5 and height > 5:  # Left click and rectangle started
            print('we are going to perform the segmentation')
            # print(self.rect)
            # print(, )
            # print( - self.x_start)
            # print( - self.y_start)
            color = (255, 255, 0)  # Blue color in BGR format
            thickness = 1
            mak_predictor.set_image(image_rgb)
            box = np.array([int(x), int(y), int(x+width), int(y+height)])
            #print(box)
            masks, scores, logits = mak_predictor.predict(box=box, multimask_output=False)
            mask_gr = masks[0, :, :] * 255
            rgb_image_mask = np.stack((mask_gr, mask_gr, mask_gr), axis=-1).astype(np.uint8)

            # rgb_image_mask = np.zeros((mask_gr.shape[0], mask_gr.shape[1], 3), dtype=np.uint8)
            # rgb_image_mask[:, :, 0] = mask_gr
            # rgb_image_mask[:, :, 1] = mask_gr
            # rgb_image_mask[:, :, 2] = mask_gr

            # Apply the mask to the original image
            result = cv2.addWeighted(original_image, 0.7, rgb_image_mask, 0.3, 0)
            #self.ax.add_patch(result[int(self.x_start):int(self.x_end), int(self.y_start):int(self.y_end), :])
            # ax.imshow()
            im.set_data(result)
            self.ax.figure.canvas.draw()
            self.rect = None
            #
            # self.x_end = self.x_start
            # self.y_end = self.y_start
            # print('one anotation')
            # plt.show()
        else:

            # self.rect = None
            # self.x_start = None
            # self.y_start = None
            # self.x_end = None
            # self.y_end = None

            #self.ax.figure.canvas.draw()
            if event.button == 1:
                original_image = np.where(rgb_image_mask > 0, 0, original_image)
                im.set_data(original_image)
                for rect in self.rectangles:
                    rect.remove()
                self.rectangles = []
                self.ax.figure.canvas.draw()
                self.rect =None
                mask_ = rgb_image_mask[:, :, 0]
                self.create_json_file(mask_)

                json_name = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/anotated_json_files/' + self.image_id + '.json'
                print(self.coco_data)
                with open(json_name, 'w') as destination_file:
                        json.dump(self.coco_data, destination_file)

                #print(self.rect)
                print('left_click')

            elif event.button == 3:
                im.set_data(original_image)
                for rect in self.rectangles:
                    rect.remove()
                self.rectangles = []
                self.ax.figure.canvas.draw()
                print('right_click')

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # res = []
        # for contour_ in res_:
        #     area = cv2.contourArea(contour_)
        #     if area > 1000:
        #         res.append(contour_)

        if hierarchy is None:  # empty mask
            return [], False

        # TODO speed this search up
        # new_polys = res.copy()
        new_polys = res[:]
        indices_to_keep = list(range(len(res)))
        for i, r1 in enumerate(res):
            for j, r2 in enumerate(res):
                if i == j:
                    continue
                if np.in1d(r2.ravel(), r1.ravel()).all():
                    if len(r2) > len(r1):
                        if i in indices_to_keep:
                            indices_to_keep.remove(i)
                    elif len(r1) > len(r2):
                        if j in indices_to_keep:
                            indices_to_keep.remove(j)

        res = [val for i, val in enumerate(new_polys) if i in indices_to_keep]

        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x for x in res if len(x) >= 6]
        return res, has_holes

    def create_json_file(self, mask_):
        res, has_hole = self.mask_to_polygons(mask_)
        # image_id = os.path.splitext(self.name)[0]
        for obj in range(len(res)):
            # random_number = random.randint(0, 5)
            polygon_sample = np.array(res[obj]).reshape((-1, 2)).astype(np.int32)
            #cv2.drawContours(back_img, [polygon_sample], 0, colors[random_number], 2)
            X = [];
            Y = []
            self.object_id += 1
            for point in polygon_sample:
                X.append(point[0]);
                Y.append(point[1])
                bbox = [int(min(X)), int(min(Y)), int(abs(max(X) - min(X))), int(abs(max(Y) - min(Y)))]
            area = abs(max(X) - min(X)) * abs(max(Y) - min(Y))
            segmentation_list = res[obj].tolist()
            unflat = self.unflatten_to_polygon(segmentation_list)
            epsilon = 0.05  # Adjust this value to control the level of simplification
            simplified_polygon = self.douglas_peucker(unflat, epsilon)
            # print(simplified_polygon)
            segmentation_list = [self.flatten_polygon(simplified_polygon)]

            self.coco_data["annotations"].append({
                    "id": int(self.object_id),
                    "image_id": self.image_id,
                    "category_id": int(1),
                    "segmentation": segmentation_list,
                    "area": int(area),
                    "bbox": bbox,
                    "iscrowd": int(0)
                })

        # imd_dic = {'file_name': self.name, 'id': image_id, 'width': int(480), 'height': int(640)}
        # self.coco_data['images'].append(imd_dic)
        # json_name = json_address + 'myjson' + str(image_id // 1000) + '.json'
        # with open(json_name, 'w') as destination_file:
        #         json.dump(coco_data, destination_file)

    def flatten_polygon(self, polygon):
        """
        Flattens a polygon represented as a list of (x, y) coordinates into a single list.

        Args:
            polygon (list): Polygon represented as a list of (x, y) coordinates.

        Returns:
            list: Flattened list of points.
        """
        flattened_points = [coord for point in polygon for coord in point]
        return flattened_points

    def unflatten_to_polygon(self, flattened_points):
        """
        Converts a flattened list of points back to a polygon represented as a list of (x, y) coordinates.

        Args:
            flattened_points (list): Flattened list of points.

        Returns:
            list: Polygon represented as a list of (x, y) coordinates.
        """
        polygon = []
        for i in range(0, len(flattened_points), 2):
            x = flattened_points[i]
            y = flattened_points[i + 1]
            polygon.append((x, y))
        return polygon

    def douglas_peucker(self, points, epsilon):
        """
        Applies the Douglas-Peucker algorithm to simplify a polygon.

        Args:
            points (list): List of (x, y) points representing the polygon.
            epsilon (float): Controls the level of simplification.

        Returns:
            list: Simplified list of (x, y) points.
        """
        dmax = 0
        index = 0

        for i in range(1, len(points) - 1):
            d = self.perpendicular_distance(points[i], points[0], points[-1])

            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            left = self.douglas_peucker(points[:index + 1], epsilon)
            right = self.douglas_peucker(points[index:], epsilon)

            return left[:-1] + right

        return [points[0], points[-1]]

    def perpendicular_distance(self, point, line_start, line_end):
        """
        Calculates the perpendicular distance between a point and a line.

        Args:
            point (tuple): (x, y) coordinates of the point.
            line_start (tuple): (x, y) coordinates of the line start point.
            line_end (tuple): (x, y) coordinates of the line end point.

        Returns:
            float: Perpendicular distance between the point and the line.
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return numerator / denominator


if __name__ == '__main__':
    image_name = "/home/fariborz_taherkhani/Downloads/coco2017/val2017/000000188296.jpg"

    original_image = cv2.imread(image_name)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    name = os.path.basename(image_name)
    image = original_image.copy()
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    im = ax.imshow(original_image, cmap='gray', vmin=0, vmax=255)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(sam)
    mak_predictor = SamPredictor(sam)

    rgb_image_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
    # Create an interactive rectangle drawer
    rectangle_drawer = InteractiveRectangleDrawer(ax, name)
    plt.show()

    #plt.show()

    #rect_selector = RectangleSelector(ax, onselect, minspanx=5, minspany=5)
    #plt.gca().add_patch(Rectangle((rectangle[0], rectangle[1]), rectangle[2] - rectangle[0], rectangle[3] - rectangle[1], fill=False,
    #              color='red'))

# Create a random plot


