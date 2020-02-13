import os
import cv2
import imutils
from Day7_SORT_tracker import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def yolo_cofiguration():

    coco_labels_path = "{base_path}/Day3_YOLO_CFG/coco.names.txt" \
                  "".format(base_path=BASE_PATH)
    labels = open(coco_labels_path).read().strip().split("\n")
    weights_path = "{base_path}/Day3_YOLO_CFG/yolov3.weights" \
                  "".format(base_path=BASE_PATH)
    cfg_path = "{base_path}/Day3_YOLO_CFG/yolov3.cfg" \
                   "".format(base_path=BASE_PATH)

    mot_tracker = Sort()

    return mot_tracker, labels, cfg_path, weights_path


def object_detection():
    mot_tracker, LABELS, cfg_path, weights_path = yolo_cofiguration()
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # determine only output layers naraceme
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    video = cv2.VideoCapture("Day3_YOLO_CFG/testing_video1.mp4")
    obj_list = ["cat", "car", "truck", "person"]
    # Some YOLO blob configuration
    confThreshold = 0.70
    nmsThreshold = 0.3

    while True:
        grabbed, image = video.read()
        if not grabbed:
            break
        image = imutils.resize(image, width=600)
        # the blob expects the RGB image, So SwapRB use to convert it into RGB.
        # 416, 416 is the image size, which is expected by YOLO.
        # 0 is the mean normalization value.
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), 0, swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        track_boxes = []
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
                # confidence rate about the detection
                if confidence > 0.93:
                    H, W = image.shape[:2]
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    # SORT object tracker expects the complete Bounding boxes
                    # therefore w+h and height+y has been added for the tracking of the detected object
                    track_boxes.append([x+20, y+20, int(width)+x+20, int(height)+y+20])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # 0.3 is NMS threshold, while 0.5 is the confidence value
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        # IDX are the number of the detected objects by YOLO.  .
        if len(idxs) > 0:
            for i in idxs.flatten():
                # rest of the objects have been ignored from the COCO dataset
                predicted_class = LABELS[classIDs[i]]
                if predicted_class not in obj_list:
                    continue
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # here I am converting the bounding boxes to numpy array,
                # which is expected by the SORT MOT
                track_boxes = np.array(track_boxes)
                trackers = mot_tracker.update(track_boxes)
                # print(len(trackers[0]))
                # exit()
                if len(trackers) > 0:
                    for track in trackers:
                        # print("extracting the track ID of the tracked object")
                        track_id = int(track[4])
                        print(track_id)
                        cropped = image[int(track[1]): int(track[3]), int(track[0]): int(track[2])]
                        cv2.imwrite("img.jpg", cropped)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "{}: {:.4f}".format(predicted_class + "" + str(track_id), confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    object_detection()
