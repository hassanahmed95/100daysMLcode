# in that project an object tracking has been perforemd.
# the object detection algorithm named as YOLO has been used for detection purpose
# dlib correlation has been used as a tracking algorithm

# TILL yet i have just performed the object detection in the video stream
# using the YOLO object detection model. . . On these detected objects, we will have to perform the
# object traking using dlib coralation traking
import os
import cv2
import numpy as np
import imutils
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
    video = cv2.VideoCapture("Day3_YOLO_CFG/testing_video1.mp4")
    while True:
        grabbed, image = video.read()
        if not grabbed:
            break
            
        image = imutils.resize(image, width=600)
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
        if len(idxs) > 0:
            for i in idxs.flatten():
                # some obects has been focused, else are rejected
                # from the list of predictions
                # only cars and trucks are detected
                obj_list = ["truck", "car"]
                predicted_class = LABELS[classIDs[i]]
                if predicted_class not in obj_list:
                    continue
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                try:
                    cropped_image = image[y:y+h, x:x+h]
                    cropped_image = cv2.resize(cropped_image, (300, 300))
                    cv2.imshow("cropped_image", cropped_image)
                except Exception:
                    continue
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    object_detection ()