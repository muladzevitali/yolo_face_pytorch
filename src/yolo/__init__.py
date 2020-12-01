from src.config import (yolo_face_config)
from .yolo import Yolo

face_detector = Yolo(yolo_face_config)
