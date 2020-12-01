from pathlib import Path

from src.yolo import face_detector

if __name__ == '__main__':
    image_path = Path('media/samples/messi.jpg')
    boxes = face_detector.detect(image_path, save=True)
