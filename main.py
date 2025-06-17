import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow

# Paramètres
video_path = "20250605_144926 -- test.mp4"
output_dir = "frames_output"
annotated_video_path = "annotated_output.mp4"
target_class_name = "Melon"
conf_threshold = 0.7

# Charger le modèle Roboflow
rf = Roboflow(api_key="PoYOulqxaReqSWTbxwT2")
project = rf.workspace().project("rap_utp_tobacco")
model = project.version(2).model

# Créer dossier de sortie
os.makedirs(output_dir, exist_ok=True)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Font (fallback)
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

frame_idx = 0
melon_total_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 5 != 0: continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    temp_path = f"{output_dir}/frame_{frame_idx:04d}.jpg"
    pil_image.save(temp_path)

    # Inference
    results = model.predict(temp_path).json()
    draw = ImageDraw.Draw(pil_image)

    frame_melon_count = 0
    
    for pred in results.get("predictions", []):
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        label = pred['class']
        conf = pred['confidence']

        if label == target_class_name and conf >= conf_threshold:
            frame_melon_count += 1
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            draw.rectangle([x1, y1, x2, y2], outline="purple", width=4)

            label_text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1 + 10, y1 - 40), label_text, font=font)
            draw.rectangle((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5), fill="purple")
            draw.text((x1 + 10, y1 - 40), label_text, fill="white", font=font)

    melon_total_count += frame_melon_count
    pil_image.save(f"{output_dir}/annotated_{frame_idx:04d}.jpg")
    frame_idx += 1

cap.release()

# Assembler les images en vidéo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

for i in range(frame_idx):
    img_path = f"{output_dir}/annotated_{i:04d}.jpg"
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()

print(f"✅ Vidéo annotée sauvegardée dans: {annotated_video_path}")
print(f"🍈 Total melons détectés (sur toutes les frames): {melon_total_count}")
