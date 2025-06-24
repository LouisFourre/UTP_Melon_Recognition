from ultralytics import YOLO
import sys, time, torch

# Parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."

# Model parameters
target_class_name = "Melon"
conf_threshold = 0.8

model = YOLO("models/detect/V2/weights/best.pt")
model.to('cuda' if torch.cuda.is_available() else 'cpu') # Need pytorch with cuda support!!

t_start = time.perf_counter()

results = model.track(video_path, show=True, conf=conf_threshold, save=True, stream=True,tracker="tracker.yaml",persist=True)

# get the last seen melon ids, cause each id is unique, last id is also the nulber of melons
last_seen_id = -1

for result in results:
    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
        max_id_in_frame = ids.max(initial=-1)
        last_seen_id = max(last_seen_id, max_id_in_frame)
        
t_stop = time.perf_counter()

# Print results
print("Number of melon: ", last_seen_id)
print("Time taken to compute: ",t_stop - t_start)
