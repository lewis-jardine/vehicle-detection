import argparse
import cv2
import numpy as np
import torch
from time import time


# construct arg parse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, help="path to input video file")
ap.add_argument('-o', '--output', type=str, default="out_file.webm", help="path to ouput video file to be created")
ap.add_argument('-c', '--confidence', type=float, default=0.2, help="objects will not be marked if confidence is below this threshold")
args = vars(ap.parse_args())


class ObjectDetect:
    """YOLO5 model implementation"""

    def __init__(self):
        """Initialises with video and model references. Gives output file name."""
        # read in video
        self.video = cv2.VideoCapture(args['input'])
        # load model from pytorch
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # will use cuda gpu if available
        self.out_file = args['output']
        self.confidence = args['confidence']


    def score_frame(self, frame):
        """Takes single frame as input, runs through yolo and returns detections with screen location"""
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1].numpy()
        cord = results.xyxyn[0][:, :-1].numpy()
        
        return labels, cord


    def class_to_label(self, x):
        """Returns a string label, given numeric label"""
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """Takes score frames results and frame, plots rectangles over identified objects with certain score"""
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            # no prediction if score is less than amount passed in arg
            if row[4] < self.confidence:
                continue

            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)

            bgr = (0, 255, 0) # rectangle colour (green)
            label_font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), label_font, 0.9, bgr, 2)
            
        return frame


    def __call__(self):
        """Called when class is executed, runs loop to read video frame by frame. Writes output to new file"""
        video = self.video
        x_shape = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))

        while True:
            start_time = time()
            ret, frame = video.read()
            assert ret # breaks loop if assert files eg. no ret so vids ended
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frame Per Second: {fps}")
            out.write(frame)


# create and execute object
od = ObjectDetect()
od()