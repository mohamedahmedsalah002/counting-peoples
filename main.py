'''import cv2

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame):
    bounding_box_cordinates, _ = HOGCV.detectMultiScale(frame, winStride=(2, 2), padding=(10, 10), scale=1.33)

    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, 'Person', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

    cv2.putText(frame, 'Status: Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {len(bounding_box_cordinates)}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)

def detect_by_camera():
    print('Opening Webcam...')
    video = cv2.VideoCapture(0)

    while True:
        check, frame = video.read()
        if not check:
            print("Couldn't open the camera.")
            break

        detect_people(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_by_camera()'''
"""import cv2
from skimage.feature import local_binary_pattern


# Function to perform human detection using traditional image processing techniques
def detect_humans(image):
    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Feature extraction - Histogram of Oriented Gradients (HOG)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)

    # Feature extraction - Local Binary Patterns (LBP)
    lbp_radius = 3
    lbp_points = 8 * lbp_radius
    lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')

    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Color Histograms
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Human Detection
    # Placeholder: Implement your human detection algorithm using the extracted features
    # For example, you can combine features or use them individually for detection

    # Post-processing
    # Placeholder: Implement post-processing techniques to refine the detection results

    # Return the detection results (for now, returning the features for demonstration)
    return hog_features, lbp, edges, hist_b, hist_g, hist_r


# Main function to capture images from the camera and perform human detection
def main():
    # Initialize the video capture object for the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform human detection on the captured frame
        hog_features, lbp, edges, hist_b, hist_g, hist_r = detect_humans(frame)

        # Display the detection results
        # For demonstration purposes, let's display the pre-processed images and extracted features
        cv2.imshow('Original', frame)
        cv2.imshow('HOG Features', hog_features)
        cv2.imshow('LBP', lbp)
        cv2.imshow('Edges', edges)
        cv2.imshow('Color Histogram (Blue)', hist_b)
        cv2.imshow('Color Histogram (Green)', hist_g)
        cv2.imshow('Color Histogram (Red)', hist_r)

        # Check if the user pressed the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()"""


'''def milliseconds_to_type(digits, num):
    # Define the time in milliseconds needed to type each digit
    digit_times = {
        '0': 200, '1': 200, '2': 200, '3': 200, '4': 200,
        '5': 200, '6': 200, '7': 200, '8': 200, '9': 200
    }

    # Check if digits contains each digit (0-9) exactly once
    if sorted(digits) != sorted('0123456789'):
        return "Error: digits should contain each digit (0-9) exactly once"

    # Check if num contains all the digits from digits
    if not all(digit in num for digit in digits):
        return "Error: num should contain all the digits from digits"

    # Initialize total time
    total_time = 0

    # Get the index of each digit in num
    index_map = {num[i]: i for i in range(len(num))}

    # Iterate through each digit in digits
    for digit in digits:
        # Get the index of the current digit in num
        index = index_map[digit]

        # Add the time for typing the current digit to the total time
        total_time += digit_times[digit] * (index + 1)

    return total_time


# Example usage:
digits = "9123456789"
num = "219"
time_needed = milliseconds_to_type(digits, num)
print("Milliseconds needed to type number {} using pattern {}: {}".format(digits, num, time_needed))'''


'''import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound
from food_facts import food_facts




def speech(text):
    print(text)
    language = "en"
    output = gTTS(text=text, lang=language, slow=False)

    output.save("./sounds/output.mp3")
    playsound("./sounds/output.mp3")


video = cv2.VideoCapture(1)
labels = []

while True:
    ret, frame = video.read()
    # Bounding box.
    # the cvlib library has learned some basic objects using object learning
    # usually it takes around 800 images for it to learn what a phone is.
    bbox, label, conf = cv.detect_common_objects(frame)

    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("Detection", output_image)

    for item in label:
        if item in labels:
            pass
        else:
            labels.append(item)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

i = 0
new_sentence = []
for label in labels:
    if i == 0:
        new_sentence.append(f"I found a {label}, and, ")
    else:
        new_sentence.append(f"a {label},")

    i += 1

speech(" ".join(new_sentence))
speech("Here are the food facts i found for these items:")

for label in labels:
    try:
        print(f"\n\t{label.title()}")
        food_facts(label)

    except:
        print("No food facts for this item")'''
'''from typing import List

def alt_subsequence_best(X: List[int]) -> int:
    count = 1
    for i in range(1, len(X)):
        if X[i] == 0 or X[i] == 1:
            if X[i] != X[i-1]:
                count += 1
    return count

# Example usage:
X1 = [0, 1, 0, 1, 0]
X2 = [0]

print("Output 1:", alt_subsequence_best(X1))  # Output: 5
print("Output 2:", alt_subsequence_best(X2))  # Output: 1'''
'''import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/mo/Downloads/caleeb.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding to obtain a binary image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the binary image
thresh = cv2.bitwise_not(thresh)

# Perform morphological operations to clean up the image
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours in the cleaned image
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours and draw rectangles around detected humans
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Human Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''import cv2
import numpy as np
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize counters
total_humans = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting objects
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # class_id 0 is for 'person'
                total_humans += 1

    # Display the number of humans detected
    cv2.putText(frame, f'Total Humans: {total_humans}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()'''

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]

area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('peoplecount1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    #    frame=cv2.flip(frame,1)
    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()