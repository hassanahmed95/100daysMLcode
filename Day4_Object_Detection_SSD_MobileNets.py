# In that project a sample project of object detection has been
# done using Deep learning framework SSD and Mobilnet
# In the following code the Caffe Version of the model has been used

# The data and code credit goes to Pyimagesearch. . .

import cv2
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_models():
    model = "{base_path}/Day4_MobileNet_SSD/MobileNetSSD_deploy.caffemodel"\
        .format(base_path= BASE_PATH)
    protext = "{base_path}/Day4_MobileNet_SSD/MobileNetSSD_deploy.prototxt.txt"\
        .format(base_path= BASE_PATH)
    return protext, model


def obj_detection(image):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    protext, model = load_models()
    net = cv2.dnn.readNetFromCaffe(protext, model)
    h,w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    net.setInput(blob)

    # here I will get all the detections results
    detections = net.forward()
    print(detections)
    print(detections.shape[2])

    for i in np.arange(0, detections.shape[2]):
        # confidence can be extracted by both methods, mentioned below
        # 0,0 are written to access the contents of inner brackets
        # 'i' shows each detected object, for each 'i' will iterate in
        # sub-list and extract relevant data from the list

        # confidence = detections[0, 0, i, 2]
        confidence = detections[0][0][i][2]

        if confidence > 0.5:
            # extract the index of the class label from the detection
            idx = int(detections[0, 0, i, 1])
            # 3: 7 are the coordinates of the bounding boxes.
            # multiplied with [w, h, w, h] in order to mold in
            # relevant form...
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("ds", image)
    cv2.waitKey(0)


if __name__ == "__main__":

    image_path = "./Day3_YOLO_CFG/soccer.jpg"
    image = cv2.imread(image_path)
    obj_detection(image)