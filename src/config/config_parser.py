"""Configuration class for applications"""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

import torch

config = configparser.ConfigParser()
config.read("config.ini")
base_dir = Path('.')

os.environ["NLS_LANG"] = "AMERICAN_AMERICA.AL32UTF8"


@dataclass()
class Media:
    media_path = base_dir.joinpath(config["MEDIA"]["MEDIA_PATH"])

    def __init__(self):
        self.media_path.mkdir(exist_ok=True, parents=True)


@dataclass()
class YoloFaceConfiguration:
    output_folder = base_dir.joinpath(config['YOLO-FACE']['OUTPUT_FOLDER'])
    batch_size = int(config['YOLO-FACE']['BATCH_SIZE'])
    conf_thresh = float(config['YOLO-FACE']['CONFIDENCE'])
    nms_thresh = float(config['YOLO-FACE']['NMS_THRESH'])
    config_path = config['YOLO-FACE']['CONFIG']
    weights_path = config['YOLO-FACE']['WEIGHTS']
    names_path = config['YOLO-FACE']['NAMES']
    image_size = int(config['YOLO-FACE']['IMAGE_SIZE'])
    scales = config['YOLO-FACE']['SCALES']
    cuda = torch.cuda.is_available()

    def __init__(self):
        self.output_folder.mkdir(exist_ok=True, parents=True)
