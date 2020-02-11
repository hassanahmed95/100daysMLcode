# in that project an object tracking has been perforemd.
# YOLO V3 has been used as the object detection model and DLIB coralation has been used
# as the object tracking algorithm
# DLIB object trracker works very well, if it is used for the single object
import os
import cv2
import numpy as np
import imutils
import dlib

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def yolo_cofiguration():

    coco_labels_path = "{base_path}/Day3_YOLO_CFG/coco.names.txt" \
                  "".format(base_path=BASE_PATH)
    labels = open(coco_labels_path).read().strip().split("\n")
    weights_path = "{base_path}/Day3_YOLO_CFG/yolov3.weights" \
                  "".format(base_path=BASE_PATH)
    cfg_path = "{base_path}/Day3_YOLO_CFG/yolov3.cfg" \
                   "".format(base_path=BASE_PATH)
    return labels, cfg_path, weights_path


def object_detection():
    LABELS, cfg_path, weights_path = yolo_cofiguration()
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # determine only output layers name
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    video = cv2.VideoCapture("Day3_YOLO_CFG/cat.mp4")
    tracker = None
    obj_list = ["person","car","cat","truck"]
    while True:
        grabbed, image = video.read()
        if not grabbed:
            break
        image = imutils.resize(image, width=600)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if tracker is None:
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                          swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []
            # layerOutputs, LABELS = object_detection(image)
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # 5 elements > centerX,centerY, width,height and score
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.95:
                        H, W = image.shape[:2]
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # 0.3 is NMS threshold, while 0.5 is the confidence value
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
            # IDX are the number of the detected objects by YOLO.  .
            if len(idxs) > 0:
                for i in idxs.flatten():
                    # rest of the objects have been ignored from the COCO dataset
                    predicted_class = LABELS[classIDs[i]]
                    if predicted_class not in obj_list:
                        continue
                    print(predicted_class)
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, (x + w), (y + h))
                    tracker.start_track(image, rect)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "{}: {:.4f}".format(predicted_class, confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
        # we have already performed the detection, let perform the tracking of the detected object  . .
        else:
            tracker.update(image)
            pos = tracker.get_position()
            # print(pos)
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(image, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    object_detection()
