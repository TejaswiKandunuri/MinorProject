import cv2
import numpy as np
from keras.models import model_from_json

# loading the model
json_file = open('AffectNet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("AffectNet.h5")
print("Loaded model from disk")

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48

labels = ['happy', 'Disgust', 'Fear', 'sad', 'angry', 'Surprise', 'neutral']

# loading the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# loading YOLO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# setting minimum probability threshold for object detection
conf_threshold = 0.9

# loading image
full_size_image = cv2.imread("image.jpeg")
if full_size_image is None:
    print("Error loading the image. Please check the image path.")
    exit(1)
print("Image Loaded")

# Get the image dimensions
image_height, image_width = full_size_image.shape[:2]

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(full_size_image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Sets the blob as the input of the network
net.setInput(blob)

# Get the names of the output layers
out_layer_names = net.getUnconnectedOutLayersNames()

# Forward pass to get output of the output layers
outs = net.forward(out_layer_names)

# Initialize the face list, confidences, and bounding boxes
faces = []
confidences = []
bounding_boxes = []

# Extract the faces and bounding boxes from the YOLO output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold and class_id == 0:  # 0 is the class_id for "person" in COCO dataset
            center_x = int(detection[0] * image_width)
            center_y = int(detection[1] * image_height)
            w = int(detection[2] * image_width)
            h = int(detection[3] * image_height)
            x = center_x - w // 2
            y = center_y - h // 2
            bounding_boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, conf_threshold, 0.4)

# Detecting emotions for each face
for i in indices:
    box = bounding_boxes[i]
    x, y, w, h = box
    roi_gray = full_size_image[y:y + h, x:x + w]
    if roi_gray.size == 0:
        print("Error: Detected face region is empty.")
        continue
    roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # predicting the emotion
    yhat = loaded_model.predict(cropped_img)
    cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    print("Emotion: " + labels[int(np.argmax(yhat))])

cv2.imshow('Emotion', full_size_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
