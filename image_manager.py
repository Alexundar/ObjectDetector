import os
import os.path as osp
import pickle
import random

import cv2

from util import *


class ImageManager:
    def __init__(self):
        self.colors = pickle.load(open("pallete", "rb"))

    def read_images(self, images):
        list_of_images = []
        try:
            list_of_images = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
        except NotADirectoryError:

            list_of_images.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print("No file or directory with the name {}".format(images))
            exit()

        loaded_images = [cv2.imread(x) for x in list_of_images]
        return loaded_images, list_of_images

    def draw_bounding_boxes(self, x, results, classes):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        image = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(self.colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(image, c1, c2, color, 7)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(image, c1, c2, color, -1)
        cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_DUPLEX, 1, [225, 255, 255], 2);
        return image

    def write_images(self, det_names, det_images):
        map(cv2.imwrite, det_names, det_images)
