from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2, os, sys, time, glob, shutil

# Parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."
output_dir = "frames_output"
annotated_video_path = "annotated_video.mp4"
os.makedirs(output_dir, exist_ok=True)

# Model parameters
target_class_name = "Melon"
conf_threshold = 0.7


try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

# Load the model from Roboflow, and not from a local file because I can't download the weights from RF
rf = Roboflow(api_key="PoYOulqxaReqSWTbxwT2")
project = rf.workspace().project("rap_utp_tobacco")
model = project.version(2).model

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
    
    results = model.predict(frame).json()
    frame_melon_count = 0
    
    draw = ImageDraw.Draw(pil_image)

    for prediction in results.get('predictions', []):
        x, y, width, height, class_conf, label = prediction['x'], prediction['y'], prediction['width'], prediction['height'], prediction['confidence'], prediction['class']
        # Calculate the bounding box coordinates in xyxy format and substracting half width and height
        # to center the box around the predicted point
        x1,y1,x2,y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2

        if label == target_class_name and class_conf >= conf_threshold:
            frame_melon_count += 1
            x1, y1 = x - width/2, y - height/2
            x2, y2 = x + width/2, y + height/2
            draw.rectangle([x1, y1, x2, y2], outline="purple", width=4)

            label_text = f"{label} {class_conf:.2f}"
            bbox = draw.textbbox((x1 + 10, y1 - 40), label_text, font=font)
            draw.rectangle((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5), fill="purple")
            draw.text((x1 + 10, y1 - 40), label_text, fill="white", font=font)

    total_melon_count += frame_melon_count
    pil_image.save(f"{output_dir}/annotated_{frame_idx:04d}.jpg")
    frame_idx += 1

cap.release()

# Create annotated video from saved frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (int(width), int(height)))

# Liste des fichiers réellement créés
annotated_images = sorted(glob.glob(f"{output_dir}/annotated_*.jpg"))

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

shutil.rmtree(output_dir)
