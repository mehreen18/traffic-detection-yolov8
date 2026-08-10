# Traffic Object Detection - YOLOv8

An end-to-end Computer Vision system that detects car, motorcycle, person, bus, and truck
using YOLOv8, trained on Google Colab, served through a FastAPI endpoint with a custom frontend.

## Project structure
- `dataset/` - images, labels, data.yaml
- `notebooks/` - training notebook
- `outputs/` - training results, predictions
- `models/` - best.pt (trained weights)
- `app/` - FastAPI inference app

## Tools used
Python, Google Colab, YOLOv8 (Ultralytics), OpenCV, FastAPI, Roboflow

## How to run
1. Load `best.pt` in `app/app.py`
2. Run with uvicorn
3. POST an image to `/predict`
