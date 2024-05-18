import os
import cv2
import time
import uuid

IMAGE_PATH = "CollectedImages"

labels = ['Hello', 'Yes', 'No', 'Thanks', 'IloveYou', 'Please']

number_of_images = 5

# Check if the directory exists
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

for label in labels:
    label_path = os.path.join(IMAGE_PATH, label)

    # Check if the label directory exists
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    # Open camera 
    cap = cv2.VideoCapture(0)
    print(f"Collecting images for {label}")
    time.sleep(3)

    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        imagename = os.path.join(label_path, f"{label}_{uuid.uuid1()}.jpg")  # Adjusted image name format
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()