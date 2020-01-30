# In that project a sample project has been done using YOLO from OpenCv.
# The code and details credit goes to Pyimagsesearch
import os
import cv2
import numpy as np
import time

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def yolo_cofiguration():
    coco_labels_path = "{base_path}/Day3_YOLO_CFG/coco.names.txt" \
                  "".format(base_path=BASE_PATH)
    labels = open(coco_labels_path).read().strip().split("\n")

    weights_path = "{base_path}/Day3_YOLO_CFG/yolov3.weights" \
                  "" .format(base_path=BASE_PATH)
    cfg_path = "{base_path}/Day3_YOLO_CFG/yolov3.cfg" \
                   "" .format(base_path=BASE_PATH)
    return labels, cfg_path, weights_path


def object_detection(image):
    LABELS, cfg_path, weights_path = yolo_cofiguration()
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # determine only output layers name
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # blobFromImage creates 4-dimensional blob from image. Optionally resize and crop
    # image from center, subtract mean values, scale values by scale factor, swap Blue and Red
    # channels.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs, LABELS


def object_recog(image):
    boxes = []
    confidences = []
    classIDs = []
    layerOutputs, LABELS = object_detection(image)
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # 5 elements > centerX,centerY, width,height and score
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.7:
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
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # passing the path of the input images
    image_path = "./Day3_YOLO_CFG/soccer.jpg"
    image = cv2.imread(image_path)
    object_recog(image)


