from inference import get_model
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Define the image file to use for inference
image_file = "20250605_144656_mp4-0050.jpg"

# Load the model from Roboflow, and not from a local file because I can't download the weights from RF
rf = Roboflow(api_key="PoYOulqxaReqSWTbxwT2")
project = rf.workspace().project("rap_utp_tobacco")
model = project.version(2).model

# Run inference to predict
results = model.predict(image_file).json()

# Filter by class "Melon" and with confidence >= 0.7
target_class_name = "Melon"
filtered_boxes = []

for prediction in results.get('predictions', []):
    x = prediction['x']
    y = prediction['y']
    width = prediction['width']
    height = prediction['height']
    class_conf = prediction['confidence']
    label = prediction['class']

    # Calculate the bounding box coordinates in xyxy format and substracting half width and height
    # to center the box around the predicted point
    x1,y1,x2,y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2

    if label == target_class_name and class_conf >= 0.7:
        filtered_boxes.append(((x1, y1, x2, y2), class_conf, label))

melon_count = len(filtered_boxes)

# Open img with PIL
with Image.open(image_file) as im: 
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 50)

    for xyxy, conf, label in filtered_boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        draw.rectangle((x1, y1, x2, y2), outline="purple", width=6)

        # Draw label and confidence above the bounding box
        # Adjust the position to avoid overlap with the bounding box
        bbox = draw.textbbox((x1 + 10, y1 - 50), f"{label} {conf:.2f}", font=font)
        draw.rectangle((bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10), fill="purple")
        draw.text((x1 + 10, y1 - 50), f"{label} {conf:.2f}", fill="white", font=font)

    # Save before displaying because the code won't save until we close the matplotlib render
    im.save("output_annotated.jpg")

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(im)
    plt.axis('off')
    plt.title(f"Melon Count: {melon_count}")
    plt.show()

# Print the outputs
print("Melon Count:", melon_count)
for _, conf, label in filtered_boxes:
    print({"label": label, "confidence": conf})
