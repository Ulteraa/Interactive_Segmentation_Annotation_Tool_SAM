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
from matplotlib.widgets import Button
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
from skimage.draw import polygon2mask
class InteractiveRectangleDrawer:
    def __init__(self, ax, name):
        self.ax = ax
        self.rect = None
        self.x_start = None
        self.y_start = None
        self.x_end = None
        self.y_end = None
        # self.rect_params = None
        self.rect = None
        self.rectangles = []
        #self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        #self.ax.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.global_onclick)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        # self.ax.figure.canvas.mpl_connect('button_release_event', self.on_points)
        self.coco_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []}

        self.rect_params = {
            'x': None,
            'y': None,
            'width': None,
            'height': None
        }
        #

        self.coco_data["categories"].append({"id": 1, "name": "package"})
        self.name = name
        self.object_id = 0
        self.image_id = os.path.splitext(self.name)[0]
        self.shape = original_image.shape
        imd_dic = {'file_name': self.name, 'id': self.image_id, 'width': int(self.shape[0]), 'height': int(self.shape[1])}
        self.coco_data['images'].append(imd_dic)

    def onclick(self, event):
        # Record the x and y coordinates of the click and display them
        global already_drawn_manual, clicks,  already_drawn_line, object_to_save_manual
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            clicks.append((x, y))

            # Draw the point on the figure
            visual_manual, = self.ax.plot(x, y, 'ro', markersize=2)
            already_drawn_manual.append(visual_manual)
            # If we have two or more points, draw lines between them

            if len(clicks) > 1:

                # Correctly remove the previous line segments, if any, to avoid duplicating lines
                while len(self.ax.lines) > 1:
                    self.ax.lines[-1].remove()  # Remove the last line artist in the list
                # # Draw lines between all successive points
                for i in range(1, len(clicks)):
                    # Get the start and end points for the current segment
                    start_point = clicks[i - 1]
                    end_point = clicks[i]
                    # Draw a line between the current and previous point
                    line, = self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')
                    already_drawn_line.append(line)
                #self.ax.plot(*zip(*clicks), marker='', color='red')
            if len(clicks)>=3:
                object_to_save_manual = 1
            self.ax.figure.canvas.draw()


    # def perform_annotation_task(self, event):
    #     global original_image
    #     original_image_ = np.where(rgb_image_mask > 0, 0, original_image)
    #     im.set_data(original_image_)
    #     for rect in self.rectangles:
    #         rect.remove()
    #     self.rectangles = []
    #     self.ax.figure.canvas.draw()
    #     self.rect = None
    #     print('left_click')

    def clear_task(self, event):
        global original_image, current_mode,  rgb_image_mask, object_to_save_point, object_to_save, point_clicks, already_drawn_points, already_drawn_manual, clicks
        if len(self.rectangles)>=1 or object_to_save:
            im.set_data(original_image)
            object_to_save = 0
            for rect in self.rectangles:
                rect.remove()
            self.rectangles = []
            self.ax.figure.canvas.draw()
        elif current_mode == 'point':
                print('we are going to section that we want to clear out the dots')
                if len(point_clicks) >= 1:
                    point_clicks.pop()
                    # Remove all point visual representations
                    # Assuming already_drawn_points is a list of matplotlib line objects for each drawn point
                    while already_drawn_points:
                        point_visual = already_drawn_points.pop()
                        point_visual.remove()
                    # Reset the list holding visual representations

                    im.set_data(original_image)
                    # already_drawn_points.remove()
                    self.ax.figure.canvas.draw()
                    if len(point_clicks):
                            print('stop')
                            for point in point_clicks:
                                #x_data_, y_data_ = zip(*point_clicks)
                                point_visual, = self.ax.plot(point[0], point[1], 'ro', markersize=2)
                                already_drawn_points.append(point_visual)
                            im.set_data(original_image)
                            self.ax.figure.canvas.draw()
                            label = [1] * len(point_clicks)
                            point = np.array(point_clicks)
                            mak_predictor.set_image(original_image)
                            masks, scores, logits = mak_predictor.predict(point_coords=point, point_labels=label,
                                                                          multimask_output=True)
                            mask_gr = masks[0, :, :] * 255
                            rgb_image_mask = np.stack((mask_gr, mask_gr, mask_gr), axis=-1).astype(np.uint8)
                            result = cv2.addWeighted(original_image, 0.7, rgb_image_mask, 0.3, 0)
                            im.set_data(result)
                            self.ax.figure.canvas.draw()
                            self.rect = None
                            # current_mode = None
                            object_to_save_point = 1
                    else:
                        object_to_save_point = 0
        elif current_mode == 'manual':
            if len(clicks) == 1:
                clicks.pop()
                point_visual_ = already_drawn_manual.pop()
                point_visual_.remove()
            elif len(clicks)>1:
                clicks.pop()
                print('check for points')
                point_visual_= already_drawn_manual.pop()
                # point_visual_.remove()
                visual_line = already_drawn_line.pop()
                visual_line.remove()


                # Reset the list holding visual representations

            im.set_data(original_image)

            self.ax.figure.canvas.draw()
            # print(clicks)
    def save_task(self, event):
        global original_image, rgb_image_mask, object_to_save, object_to_save_point, point_clicks, current_mode, already_drawn_points, clicks, object_to_save_manual, already_drawn_manual, already_drawn_line
        current_mode = None
        print('here we are going to save')
        if object_to_save:
            mask_ = rgb_image_mask[:, :, 0]
            self.create_json_file(mask_)
            json_name = 'anotated_json_files/' + self.image_id + '.json'
            # print(self.coco_data)
            with open(json_name, 'w') as destination_file:
                json.dump(self.coco_data, destination_file)
            # global original_image
            original_image = np.where(rgb_image_mask > 0, 0, original_image)
            im.set_data(original_image)
            for rect in self.rectangles:
                rect.remove()
            self.rectangles = []
            self.ax.figure.canvas.draw()
            self.rect = None
            rgb_image_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
            object_to_save = 0
            print('saved!')
        elif object_to_save_point == 1:
            print('we are here')
            mask_ = rgb_image_mask[:, :, 0]
            self.create_json_file(mask_)
            json_name = 'anotated_json_files/' + self.image_id + '.json'
            # print(self.coco_data)
            with open(json_name, 'w') as destination_file:
                json.dump(self.coco_data, destination_file)
            # global original_image
            original_image = np.where(rgb_image_mask > 0, 0, original_image)
            im.set_data(original_image)
            for rect in self.rectangles:
                rect.remove()
            self.rectangles = []
            while already_drawn_points:
                point_visual = already_drawn_points.pop()
                point_visual.remove()
            self.ax.figure.canvas.draw()
            self.rect = None
            rgb_image_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
            object_to_save_point = 0
            point_clicks = []
            already_drawn_points = []
            print('saved!')
        elif object_to_save_manual:
            for point_visual in already_drawn_manual:
                try:
                    point_visual.remove()
                except ValueError as e:
                    print(f"Error removing point_visual: {e}")
            already_drawn_manual.clear()  # Clear the list after attempting to remove all

            # Attempt to remove all lines
            for line_visual in already_drawn_line:
                try:
                    line_visual.remove()
                except ValueError as e:
                    print(f"Error removing line_visual: {e}")
            already_drawn_line.clear()
            already_drawn_line = []
            already_drawn_manual = []
            height, width = original_image.shape[:2]
            mask_ = np.zeros((height, width), dtype=np.uint8)
            points = np.array(clicks)
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask_, [points], 255)
            self.create_json_file(mask_)
            json_name = 'anotated_json_files/' + self.image_id + '.json'
            # print(self.coco_data)
            with open(json_name, 'w') as destination_file:
                json.dump(self.coco_data, destination_file)
            # global original_image
            print('saved!')
            object_to_save_manual = 0
            clicks = []
            rgb_image_mask = np.stack((mask_, mask_, mask_), axis=-1).astype(np.uint8)
            original_image = np.where(rgb_image_mask > 0, 0, original_image)


            im.set_data(original_image)
            self.ax.figure.canvas.draw()
        elif len(clicks)<3:
            for point_visual in already_drawn_manual:
                try:
                    point_visual.remove()
                except ValueError as e:
                    print(f"Error removing point_visual: {e}")
            already_drawn_manual.clear()  # Clear the list after attempting to remove all

            # Attempt to remove all lines
            for line_visual in already_drawn_line:
                try:
                    line_visual.remove()
                except ValueError as e:
                    print(f"Error removing line_visual: {e}")
            already_drawn_line.clear()
            already_drawn_line = []
            already_drawn_manual = []
            clicks = []
            im.set_data(original_image)
            self.ax.figure.canvas.draw()
    def on_press(self, event):
        if event.button == 1 and (event.xdata is not None) and (event.ydata is not None):  # Left click to start drawing
            # print(event.xdata, event.ydata)
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

    def global_onclick(self, event):
        global current_mode
        if current_mode == 'manual':
            self.onclick(event)
        elif current_mode == 'box':
            self.on_press(event)
        elif current_mode == 'point':
            print('current mode is point')
            # pass
        #     self.on_points(event)
    #
    # def set_function1(self, event):
    #     global current_mode
    #     current_mode = 'function1'
    #     print("Mode set to Function 1")
    #
    # def set_function2(self, event):
    #     global current_mode
    #     current_mode = 'function2'
    #     print("Mode set to Function 2")

    def on_motion(self, event):
        global current_mode
        if (current_mode == 'box') and (self.rect is not None) and (event.button == 1) and (event.xdata is not None) and (event.ydata is not None) and (self.x_start is not None) and (self.y_start is not None):  # Left click and rectangle started
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
    def on_undo(self,event):
        global image, original_image, im, num_object_per_annotation
        annotations = self.coco_data['annotations']
        # print(annotations)
        if annotations:
            print('we are going to undo')
            for iteration in range(num_object_per_annotation):
              if annotations:
                instance = annotations.pop()
                segmentation = instance['segmentation']
                image_height, image_width, _ = image.shape

                polygon_flipped = np.array(segmentation[0]).reshape((-1, 2))
                polygon = np.array([[y, x] for x, y in polygon_flipped])

                # Create a binary mask from the polygon
                mask = polygon2mask((image_height, image_width), polygon)
                # has_true_value = np.any(mask)
                # print(has_true_value)
                rgb_image_mask = np.stack((mask, mask, mask), axis=-1).astype(np.uint8)

                original_image = np.where(rgb_image_mask > 0, image, original_image)
            im.set_data(original_image)
            self.ax.figure.canvas.draw()

            # print(segmentation)
        else:
            print('it just went there!')

    def on_brush(self, event):
        global image, original_image
        x = int(self.rect_params['x']);
        y = int(self.rect_params['y']);
        width = int(self.rect_params['width']);
        height = int(self.rect_params['height'])
        image_height, image_width = image.shape[:2]
        mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        mask[y:y+height, x:x+width, :] = 1
        original_image = np.where(mask> 0, image, original_image)
        for rect in self.rectangles:
            rect.remove()
        self.rectangles = []
        im.set_data(original_image)
        self.ax.figure.canvas.draw()
    def on_release(self, event):

        global rgb_image_mask, original_image, current_mode, object_to_save
        if current_mode == 'box' and self.rect is not None:
            x = self.rect_params['x'];
            y = self.rect_params['y'];
            width = self.rect_params['width'];
            height = self.rect_params['height']

            if width > 5 and height > 5:  # Left click and rectangle started
                #print('we are going to perform the segmentation')
                color = (255, 255, 0)  # Blue color in BGR format
                thickness = 1
                mak_predictor.set_image(original_image)
                box = np.array([int(x), int(y), int(x+width), int(y+height)])
                masks, scores, logits = mak_predictor.predict(box=box, multimask_output=False)
                mask_gr = masks[0, :, :] * 255
                rgb_image_mask = np.stack((mask_gr, mask_gr, mask_gr), axis=-1).astype(np.uint8)
                result = cv2.addWeighted(original_image, 0.7, rgb_image_mask, 0.3, 0)
                im.set_data(result)
                self.ax.figure.canvas.draw()
                self.rect = None
                current_mode = None
                object_to_save = 1
        # elif current_mode == 'save' and object_to_save:
        #     print('there is somethig.....')
        #     self.save_task(event)
        #     current_mode = None
        #     object_to_save = 0
        elif current_mode == 'point' and event.inaxes == self.ax:
            print('again we came to points-----')
            # print(event.ydata, event.xdata)
            self.on_points(event)
        # elif current_mode == 'brush':
        #     self.on_brush(event)

        # elif current_mode == 'undo':
        #     print('again we came to points for undo-----')
        #
        #     self.on_undo(event)
        else:
            print('it went here')
            # current_mode = None
            # object_to_save = 0

    def on_points(self, event):
        # x = self.rect_params['x'];
        # y = self.rect_params['y'];
        # width = self.rect_params['width'];
        # height = self.rect_params['height']
        global rgb_image_mask, original_image, current_mode, object_to_save_point, already_drawn_points
        if (event.xdata is not None) and (event.ydata is not None):
            x, y = event.xdata, event.ydata
            point_clicks.append([x, y])
            # Draw the point on the figure
            point_visual, = self.ax.plot(x, y, 'ro', markersize=2)
            already_drawn_points.append(point_visual)
            # print(already_drawn_points)
            label = [1] * len(point_clicks)
            point = np.array(point_clicks)
            mak_predictor.set_image(original_image)
            masks, scores, logits = mak_predictor.predict(point_coords=point, point_labels=label, multimask_output=True)
            # if self.rect is not None and width > 5 and height > 5:  # Left click and rectangle started
            #     # print('we are going to perform the segmentation')
            #     color = (255, 255, 0)  # Blue color in BGR format
            #     thickness = 1
            #     mak_predictor.set_image(image_rgb)
            #     box = np.array([int(x), int(y), int(x + width), int(y + height)])
            #     masks, scores, logits = mak_predictor.predict(box=box, multimask_output=False)
            mask_gr = masks[0, :, :] * 255
            rgb_image_mask = np.stack((mask_gr, mask_gr, mask_gr), axis=-1).astype(np.uint8)
            result = cv2.addWeighted(original_image, 0.7, rgb_image_mask, 0.3, 0)
            im.set_data(result)
            self.ax.figure.canvas.draw()
            self.rect = None
            # current_mode = None
            object_to_save_point = 1
            # elif current_mode == 'save' and object_to_save:
            #     print('there is somethig.....')
            #     self.save_task(event)
            #     current_mode = None
            #     object_to_save = 0
            # else:
            #     print('it went here')

        # else:
        #
        #     annotate_button.on_clicked(self.perform_annotation_task)
        #     # if event.button == 1:
        #     #     original_image = np.where(rgb_image_mask > 0, 0, original_image)
        #     #     im.set_data(original_image)
        #     #     for rect in self.rectangles:
        #     #         rect.remove()
        #     #     self.rectangles = []
        #     #     self.ax.figure.canvas.draw()
        #     #     self.rect =None
        #     #     mask_ = rgb_image_mask[:, :, 0]
        #     #
        #     #     self.create_json_file(mask_)
        #     #
        #     #     json_name = '/home/fariborz_taherkhani/SwinT_detectron2-main_one_package_freeze_second_v/anotated_json_files/' + self.image_id + '.json'
        #     #     print(self.coco_data)
        #     #     with open(json_name, 'w') as destination_file:
        #     #             json.dump(self.coco_data, destination_file)
        #     #
        #     #     #print(self.rect)
        #     #     print('left_click')
        #
        #     clear_button.on_clicked(self.clear_task)
        #
        #     save_button.on_clicked(self.save_task)

            # if event.button == 3:
            #     im.set_data(original_image)
            #     for rect in self.rectangles:
            #         rect.remove()
            #     self.rectangles = []
            #     self.ax.figure.canvas.draw()
            #     print('right_click')

    def mask_to_polygons(self, mask):

        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None:  # empty mask
            return [], False


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
        global  num_object_per_annotation
        res, has_hole = self.mask_to_polygons(mask_)
        num_object_per_annotation = len(res)
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

def set_box(event):
    global current_mode
    current_mode = 'box'
    print("Mode set to box")
def set_point(event):
    global current_mode
    current_mode = 'point'
    print("Mode set to point")
def set_manual(event):
    global current_mode
    current_mode = 'manual'
    print("Mode set to manual")

def set_save(event):
    global current_mode
    current_mode = 'save'
    print("Mode set to save")

def set_clear(event):
    global current_mode
    current_mode = 'clear'
    print("Mode set to clear")
def set_save(event):
    global current_mode
    current_mode = 'save'
    print("Mode set to save")
def set_undo(event):
    global current_mode
    current_mode = 'undo'
    print("Mode set to undo")
def set_brush(event):
    global current_mode
    current_mode = 'brush'
    print("Mode set to brush")
if __name__ == '__main__':
    # # Add this at the end of your main block, after initializing the InteractiveRectangleDrawer
    #
    # # Adjust the main figure to make room for the buttons
    # plt.subplots_adjust(bottom=0.3)
    #
    # # Create a button for annotation
    # annotate_ax = plt.axes([0.7, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
    # annotate_button = Button(annotate_ax, 'Annotate')
    # # annotate_button.on_clicked(rectangle_drawer.on_release)  # Assuming on_release does the annotation
    #
    # # Create a button for clearing annotations
    # clear_ax = plt.axes([0.81, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
    # clear_button = Button(clear_ax, 'Clear')
    # # clear_button.on_clicked(rectangle_drawer.clear_annotations)
    im_folder_path = '/home/fariborz_taherkhani/Downloads/blister_pack_single'
    for filename in os.listdir(im_folder_path ):
            image_name = os.path.join(im_folder_path,filename)
            original_image = cv2.imread(image_name)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            name = os.path.basename(image_name)
            image = original_image.copy()
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots()
            clicks = []
            point_clicks =[]
            current_mode = None
            object_to_save = 0
            object_to_save_point = 0
            object_to_save_manual = 0
            already_drawn_points = []
            already_drawn_manual = []
            already_drawn_line = []
            num_object_per_annotation = 0
            im = ax.imshow(original_image, cmap='gray', vmin=0, vmax=255)
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = "vit_h"
            CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
            sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
            sam.to(device=DEVICE)

            mask_generator = SamAutomaticMaskGenerator(sam)
            mak_predictor = SamPredictor(sam)

            rgb_image_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)

            rectangle_drawer = InteractiveRectangleDrawer(ax, name)
            plt.subplots_adjust(bottom=0.25)


            annotate_ax = plt.axes([0.35, 0.05, 0.1, 0.075])  # Example coordinates
            annotate_button = Button(annotate_ax, 'box')
            annotate_button.on_clicked(set_box)

            point_ax = plt.axes([0.46, 0.05, 0.1, 0.075])  # Example coordinates
            point_button = Button(point_ax, 'point')
            point_button.on_clicked(set_point)


            clear_ax = plt.axes([0.57, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
            clear_button = Button(clear_ax, 'clear')
            # clear_button.on_clicked(set_clear)
            clear_button.on_clicked(rectangle_drawer.clear_task)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            save_ax = plt.axes([0.68, 0.05, 0.1, 0.075])
            save_button = Button(save_ax, 'save')
            save_button.on_clicked(set_save)
            save_button.on_clicked(rectangle_drawer.save_task)

            manual_ax = plt.axes([0.24, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
            manual_button = Button(manual_ax, 'manual')
            manual_button.on_clicked(set_manual)

            undo_ax = plt.axes([0.13, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
            undo_button = Button(undo_ax, 'undo')
            undo_button.on_clicked(set_undo)
            undo_button.on_clicked(rectangle_drawer.on_undo)

            brush_ax = plt.axes([0.79, 0.05, 0.1, 0.075])
            brush_button = Button(brush_ax, 'brush')
            brush_button.on_clicked(set_brush)
            brush_button.on_clicked(rectangle_drawer.on_brush)
            # save_button.on_clicked(rectangle_drawer.save_task)


            # clear_button.on_clicked(rectangle_drawer.clear_annotations)
            # plt.subplots_adjust(bottom=0.3)
            #
            # # Create a button for annotation
            # annotate_ax = plt.axes([0.7, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
            # annotate_button = Button(annotate_ax, 'Annotate')
            # # annotate_button.on_clicked(rectangle_drawer.on_release)  # Assuming on_release does the annotation
            #
            # # Create a button for clearing annotations
            # clear_ax = plt.axes([0.81, 0.05, 0.1, 0.075])  # Adjust these coordinates/sizes as needed
            # clear_button = Button(clear_ax, 'Clear')
            # # Create an interactive rectangle drawer
            plt.show()



