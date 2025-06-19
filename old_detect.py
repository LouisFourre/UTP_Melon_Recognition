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


try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

model = YOLO("models/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_melon_count=0
frame_idx = 0

t_start = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # converting the frame to PIL format so we can draw on it later 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    print(f"Processing frame {frame_idx}")
    
    results = model.predict(frame)

    frame_melon_count = 0
    draw = ImageDraw.Draw(pil_image)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]

        if label == target_class_name and conf >= conf_threshold:
            frame_melon_count += 1
            draw.rectangle([x1, y1, x2, y2], outline="purple", width=4)

            label_text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1 + 10, y1 - 40), label_text, font=font)
            draw.rectangle((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5), fill="purple")
            draw.text((x1 + 10, y1 - 40), label_text, fill="white", font=font)

    total_melon_count += frame_melon_count
    pil_image.save(f"{annotated_dir}/annotated_{frame_idx:04d}.jpg")
    frame_idx += 1

cap.release()

# Create annotated video from saved frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (int(width), int(height)))

# Liste des fichiers réellement créés
annotated_images = sorted(glob.glob(f"{annotated_dir}/annotated_*.jpg"))

for img_path in annotated_images:
    frame = cv2.imread(img_path)
    if frame is not None:
        out.write(frame)
    else:
        print(f"Warning: Could not read {img_path}")

t_stop = time.perf_counter()
time_taken = (t_stop - t_start)

out.release()

# Print the outputs
print("Melon Count:", total_melon_count)
print("Time taken (seconds):", time_taken)

shutil.rmtree(annotated_dir)  # Clean up the annotated frames directory
