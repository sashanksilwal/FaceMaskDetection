# CNN Transfer Learning for Mask Detection

Lightweight face-mask detection using transfer learning (MobileNetV2 / VGG16 examples).  
Fast inference with OpenCV DNN for face detection + a small TensorFlow classifier for mask vs no-mask.

![Training loss and accuracy](outputs/plots/plotMobileNet.png)
![VGG16](outputs/plots/plotVGG.png)

---
## Motivation

During COVID, our campus had a camera system that flagged mask violations, but it was slow and unreliable.  
This project is a quick experiment to build a fast, deployable detector: using OpenCV’s DNN for face localization and a lightweight TensorFlow `MobileNetV2` model for mask detection.  
The goal: portable, low-latency inference and reproducible results even on modest hardware.

---
## Quick start

> Recommended: use Python **3.10** (TensorFlow 2.15 + NumPy 1.26 compatibility)

```bash
# create and activate venv
/Users/you/anaconda3/bin/python3.10 -m venv ./facemask
source ./facemask/bin/activate

# upgrade pip build tools
pip install --upgrade pip setuptools wheel

# install dependencies
pip install -r requirements.txt

# Run the detector:
python src/detect_mask_video.py
```
---

## Contributing

To extend this repo:

- Add new model checkpoints in `models/`
- Create a Flask API under `server/`
- Dockerize with `Dockerfile` + `docker-compose.yml`

Pull requests are welcome.

---

## License

MIT License (or your preferred license).

---




