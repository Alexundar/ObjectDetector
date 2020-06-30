import pickle
from abc import ABC, abstractmethod


class AbstractImageManager(ABC):
    def __init__(self):
        super().__init__()
        self.colors = pickle.load(open("../pallete", "rb"))

    @abstractmethod
    def read_images(self, images):
        pass

    @abstractmethod
    def draw_bounding_boxes(self, x, results, classes):
        pass

    @abstractmethod
    def write_images(self, image_names, images):
        pass
