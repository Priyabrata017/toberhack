import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
#activate vidgear
from vidgear.gears import VideoGear
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import time
from deep_sort import DeepSort
from utils.draw import draw_boxes
deepsort = DeepSort("ckpt.t7")
model = 'E:\My work\Files\projects\detectron2_webapp_trained_on_custom_dataset\config3.yml'
cfg = get_cfg()
cfg.merge_from_file("E:\My work\Files\projects\detectron2_webapp_trained_on_custom_dataset\config3.yml")

cfg.MODEL.WEIGHTS = "E:\My work\Files\projects\detectron2_webapp_trained_on_custom_dataset\model_final3.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model

MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes =['all ok', 'helmet & reflective jacket missing', 'helmet & shoes missing', 'helmet missing', 'no protection', 'reflective jacket missing', 'shoes & reflective jacket missing', 'shoes missing']

#video_capture = Videogear(source="E:/My work/Files/projects/deep_sort_yolo/test1.mp4").start()
video_capture = VideoGear(0).start()
#video_capture = cv2.VideoCapture(0)

# if video_capture.open('video.mp4'):
#width, height = int(video_capture.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = video_capture.stream.get(cv2.CAP_PROP_FPS)
#video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
#cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("test", 800,600)
width,height=100,100

while True:
    frame = video_capture.read()
    if frame is None:
        break
    start = time.time()
    xmin, ymin, xmax, ymax = 0, 0, width, height
    im = frame[ymin:ymax, xmin:xmax, (2, 1, 0)]
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('', v.get_image()[:, :, ::-1])
    #print(outputs["instances"].pred_boxes, outputs["instances"].scores, outputs["instances"].pred_classes)

    bbox_xywh=outputs["instances"].pred_boxes.tensor.cpu().numpy()
    cls_conf=outputs["instances"].scores.cpu().numpy()
    cls_ids=outputs["instances"].pred_classes.cpu().numpy()
    if bbox_xywh is not None:
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:, 3] *= 1.2
        cls_conf = cls_conf[mask]
        outputs = deepsort.update(bbox_xywh, cls_conf, im)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            frame = draw_boxes(frame, bbox_xyxy, identities, offset=(xmin, ymin))
            cv2.imshow('', frame)
            print(identities)
    key=cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

# safely close video stream
video_capture.stop()

model_url = 'E:\My work\Files\projects\detectron2_webapp_trained_on_custom_dataset\model_final3.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()