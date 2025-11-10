# =============================================
# Face Mask Detection - Modernized (TF 2.x + OpenCV 4.x)
# =============================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pathlib
import yaml

# Prevent MKL/OpenMP duplicate loading warnings on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



# -------------------------------------------------
# Load configuration and initialize directories
# -------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# Default paths if config.yaml doesn't exist
default_config = {
    "paths": {
        "face_detector_dir": "models/face_detector",
        "mask_model": "models/mask_detector.model",
        "results_dir": "outputs/results"
    }
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = default_config

FACE_DETECTOR_DIR = BASE_DIR / config["paths"]["face_detector_dir"]
MASK_MODEL_PATH = BASE_DIR / config["paths"]["mask_model"]
RESULTS_DIR = BASE_DIR / config["paths"]["results_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Load pre-trained face detector and mask classifier
# -------------------------------------------------
print("[INFO] loading face detector model...")
prototxtPath = FACE_DETECTOR_DIR / "deploy.prototxt"
weightsPath = FACE_DETECTOR_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(str(prototxtPath), str(weightsPath))

print("[INFO] loading face mask detector model...")
maskNet = load_model(str(MASK_MODEL_PATH))


# -------------------------------------------------
# Function: detect and predict mask
# -------------------------------------------------
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (224, 224), (104.0, 177.0, 123.0)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Bound box limits
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract, preprocess face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Predict masks for all detected faces
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32, verbose=0)

    return (locs, preds)


# -------------------------------------------------
# Start real-time video stream
# -------------------------------------------------
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)  # Camera warm-up

# Optional: record output
output_path = str(RESULTS_DIR / "mask_detection_output.avi")
writer = None

while True:
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=640)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Draw predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(
            frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
        )
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Initialize writer if recording
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    cv2.imshow("Face Mask Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
if writer is not None:
    writer.release()

print(f"[INFO] video saved to: {output_path}")
