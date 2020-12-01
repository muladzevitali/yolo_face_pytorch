from pathlib import Path
from typing import Union

import cv2
import numpy
import torch

from .datasets import (pad_to_square, transforms, resize)
from .models import Darknet
from .utils import (load_classes, non_max_suppression, rescale_boxes)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class YoloConfig:
    image_folder = 'data/samples'
    config_path = 'media/checkpoints/yolo/yolo_face.cfg'
    weights_path = 'media/checkpoints/yolo/yolo_face.weights'
    names_path = 'media/checkpoints/yolo/yolo_face.names'
    output_folder = Path('media/outputs')
    conf_thresh = 0.8
    nms_thresh = 0.4
    batch_size = 1
    n_cpu = 0
    image_size = 416
    checkpoint_model = ''


class Yolo(Darknet):
    def __init__(self, config):
        super().__init__(config.config_path, config.image_size)
        self.config = config

        if config.weights_path.endswith(".weights"):
            # Load darknet weights
            self.load_darknet_weights(config.weights_path)
        else:
            # Load checkpoint weights
            self.load_state_dict(torch.load(config.weights_path, map_location=device))
        self.classes = load_classes(config.names_path)
        self.to(device)
        self.eval()

    def detect(self, original_image: Union[Path, str, numpy.array], save: bool = False):
        """
        Detect desired objects within the original image
        :param original_image:
        :param save: flag whether to save output with bounding boxes
        return bounding boxes of type [[bottom_left, top_right, class], ..]
        """
        image_tensor, original_image = self.load_image(original_image)

        with torch.no_grad():
            detections = self(image_tensor)
            detections = non_max_suppression(detections, self.config.conf_thresh, self.config.nms_thresh)[0]

        detections = rescale_boxes(detections, self.config.image_size, original_image.shape[:2])
        detections = self.prettify_boxes(detections)

        if save:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            self.draw_rectangles(detections, original_image, self.classes, output_folder=self.config.output_folder)

        return detections

    def load_image(self, original_image: Union[Path, numpy.array]):
        """
        Load image for model input.
        :param original_image: Path or numpy image in bgr format(read from cv2)
        :return tensor transformed image and original numpy array of image
        """
        if isinstance(original_image, Path) or isinstance(original_image, str):
            original_image = cv2.imread(str(original_image))

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(original_image)
        image, _ = pad_to_square(image, 0)
        image = resize(image, self.config.image_size).unsqueeze(0).to(device)

        return image, original_image

    @staticmethod
    def prettify_boxes(detections):
        boxes = list()

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            bottom_left = (int(x1), int(y1))
            top_right = (int(x2), int(y2))
            cls = int(cls_pred)
            boxes.append([bottom_left, top_right, cls])

        return boxes

    @staticmethod
    def draw_rectangles(detections, image, classes, output_folder, path=None) -> None:
        """
        Draw rectangle over objects in image
        """
        for bottom_left, top_right, cls in detections:
            label = classes[cls]
            image = cv2.rectangle(image, bottom_left, top_right, (255, 0, 0), 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            text_top_right = bottom_left[0] + t_size[0] + 3, bottom_left[1] + t_size[1] + 4
            image = cv2.rectangle(image, bottom_left, text_top_right, (255, 0, 0), -1)
            image = cv2.putText(image, label, (bottom_left[0], bottom_left[1] + t_size[1] + 4),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                [225, 255, 255], 1)

        image_name = path.name if path else 'output.jpg'
        image_path = output_folder.joinpath(image_name)

        cv2.imwrite(str(image_path), image)

        return
