import pyttsx3
import cv2
import numpy as np
import os


classNames = []
classFile = "./config_files/coco.names"  


with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = './config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = './config_files/frozen_inference_graph.pb'


if not os.path.exists(configPath) or not os.path.exists(weightsPath):
    print("Model files not found.")
    exit()

# Create the network
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Check if network was created successfully
if net is None:
    print("Failed to create the network.")
    exit()

# These are some suggested settings from the tutorial, others are fine but this can be used as a baseline
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start Webcam (using laptop's built-in camera, index 0)
cap = cv2.VideoCapture(0)
# Initialize pyttsx3 engine
engine = pyttsx3.init()

while True:
    success, image = cap.read()

    if not success:
        print("Failed to capture image from webcam.")
        break

    # Tuple unpacking net.detect provides ID of object, confidence and bounding box
    classIds, confs, bbox = net.detect(image, confThreshold=0.45)

    # Extract co-ordinates of bounding box (with NMS)
    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Speak the name of the detected object
            object_name = classNames[classId - 1]
            engine.say(object_name)
            engine.runAndWait()

    # Show output
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
