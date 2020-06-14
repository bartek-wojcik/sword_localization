import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
args = vars(ap.parse_args())
# load the COCO class labels our YOLO model was trained on
script_dir = os.path.dirname(os.path.abspath(__file__))
# initialize a list of colors to represent each possible class label
weights_path = os.path.sep.join([script_dir, "yolov3.weights"])
config_path = os.path.sep.join([script_dir, "yolov3-custom.cfg"])

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
index = 1

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    frame = {}
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.1:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                if classID not in frame:
                    frame[classID] = (centerX, centerY)
                if len(frame.keys()) > 1:
                    break

    line = ''
    for i in range(2):
        if i in frame:
            line += str(frame[i][0]) + ',' + str(frame[i][1])
        else:
            line += '#,#'
        if i == 0:
            line += ','
    print(line)
    index += 1

vs.release()
