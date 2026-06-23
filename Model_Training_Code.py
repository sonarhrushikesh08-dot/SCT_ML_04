import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

DATASET_PATH = r"C:\Users\Administrator\Desktop\LeapGestRecog"
IMG_SIZE = 64

images = []
labels = []

gesture_names = {}
label_id = 0

persons = sorted(os.listdir(DATASET_PATH))

for person in persons:
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    gestures = sorted(os.listdir(person_path))

    for gesture_folder in gestures:
        gesture_path = os.path.join(person_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        # labeling gestures based on folder names
        gesture_name = gesture_folder.lower()

        if gesture_name not in gesture_names:
            gesture_names[gesture_name] = label_id
            label_id += 1

        label = gesture_names[gesture_name]

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

num_classes = len(gesture_names)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential([
    Input(shape=(64,64,3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10
)

model.save("gesture_model.h5")

with open("gesture_labels.json", "w") as f:
    json.dump(gesture_names, f)

print("Model trained successfully")