import cv2
import numpy as np
import json
import time
import pyautogui
from tensorflow.keras.models import load_model
from collections import deque

# Loading the  trained model
model = load_model("gesture_model.h5")

# Loading the label map
with open("gesture_labels.json", "r") as f:
    labels = json.load(f)

# reverse mapping of index --> gesture name
labels = {v: k for k, v in labels.items()}

gesture_response = {
    "palm": "STOP",
    "l": "SELECT MODE",
    "fist": "EXIT",
    "fist_moved": "DRAG MODE",
    "thumb": "VOLUME UP",
    "index": "CURSOR CONTROL",
    "ok": "CONFIRM CLICK",
    "palm_moved": "SCROLL MODE",
    "c": "CANCEL / BACK",
    "down": "VOLUME DOWN"
}

IMG_SIZE = 64
cap = cv2.VideoCapture(0)

history = deque(maxlen=7)
last_action_time = 0
cooldown = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box [hand region]
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    class_id = np.argmax(pred)
    confidence = np.max(pred)

    gesture = labels[class_id].lower()

    # for smoothing
    history.append(gesture)
    gesture = max(set(history), key=history.count)

    action = gesture_response.get(gesture, "UNKNOWN")

    # for display
    cv2.putText(frame,
                f"{gesture} -> {action} ({confidence:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    now = time.time()

    #  for triggering an action only if the confidence is high and a cooldown period has passed
    if confidence > 0.85 and (now - last_action_time) > cooldown:

        if gesture == "palm":
            pyautogui.press("space")

        elif gesture == "fist":
            pyautogui.press("esc")

        elif gesture == "l":
            pyautogui.press("tab")

        elif gesture == "ok":
            pyautogui.click()

        elif gesture == "thumb":
            pyautogui.press("volumeup")

        elif gesture == "down":
            pyautogui.press("volumedown")

        last_action_time = now

    cv2.imshow("Gesture Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()