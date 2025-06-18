from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2, os, sys, time, glob, shutil

# Parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."
annotated_dir = "frames_output"
annotated_video_path = "output/annotated_video.mp4"
os.makedirs(annotated_dir, exist_ok=True)

# Model parameters
target_class_name = "Melon"
conf_threshold = 0.7

model = YOLO("models/detect/train2/weights/best.pt")

results = model.track(video_path,show=True, conf=conf_threshold)