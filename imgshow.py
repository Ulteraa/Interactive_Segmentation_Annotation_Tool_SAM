import  matplotlib as plt
import  os
import cv2
import  time
saved_path='/home/fariborz_taherkhani/train_resized/images'
folder_path = '/home/fariborz_taherkhani/train/images'
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image using OpenCV
        image_path = os.path.join(folder_path, filename)
        # print(filename)
        # start_time = time.time()
        im = cv2.imread(image_path)
        cv2.imshow(filename, im)
        cv2.waitKey(0)
        F=0

        #resized = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)
        # end_times=time.time()
        # print(end_times-start_time)
        #saved_image_path = os.path.join(saved_path, filename)
        # print(im.shape)
        #cv2.imwrite(saved_image_path, resized)


