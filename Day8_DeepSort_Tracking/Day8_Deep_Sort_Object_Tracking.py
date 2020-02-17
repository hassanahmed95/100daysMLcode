import os
import cv2
import imutils
import numpy as np
# here I am importing the files for the Deep Sort Tracking Algorithm
import sys
sys.path.insert(1, "deep_sort/")
import warnings
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def yolo_cofiguration():

    coco_labels_path = "{base_path}/YOLO_CFG/coco.names.txt" \
                  "".format(base_path=BASE_PATH)
    labels = open(coco_labels_path).read().strip().split("\n")
    weights_path = "{base_path}/YOLO_CFG/yolov3.weights" \
                  "".format(base_path=BASE_PATH)
    cfg_path = "{base_path}/YOLO_CFG/yolov3.cfg" \
                   "".format(base_path=BASE_PATH)
    return  labels, cfg_path, weights_path


def object_detection():
    LABELS, cfg_path, weights_path = yolo_cofiguration()
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # determine only output layers naraceme
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    video = cv2.VideoCapture("YOLO_CFG/Copy of 00099.MTS")
    # obj_list = ["cat", "car", "truck", "person"]
    obj_list = ["car", "truck"]
    # Some YOLO blob configuratqion
    confThreshold = 0.70
    nmsThreshold = 0.3

    # here I am importing the files for the Deep Sort Tracking
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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
                if confidence > 0.95:
                    H, W = image.shape[:2]
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    # SORT object tracker expects the complete Bounding boxes
                    # therefore w+h and height+y has been added for the tracking of the detected object
                    track_boxes.append([x+20, y+20, int(width)+20, int(height)+20])
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

                text = "{}: {:.4f}".format(predicted_class + "", confidences[i])
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # track_boxes = np.array(track_boxes)
                features = encoder(image, track_boxes)

                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(track_boxes, features)]
                boxs = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxs, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                # here I will go through from the object tracker
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 0, 255), 2)
                    # print(track.track_id)
                    # cv2.putText(image, str(track.track_id), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)

                # for det in detections:
                #     bbox = det.to_tlbr()
                #     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # text = "{}: {:.4f}".format(predicted_class + "" , confidences[i])
                # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    object_detection()