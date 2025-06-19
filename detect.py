from ultralytics import YOLO
import sys, time, torch

# Parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."

# Model parameters
target_class_name = "Melon"
conf_threshold = 0.7

model = YOLO("models/detect/train2/weights/best.pt")
model.to('cuda' if torch.cuda.is_available() else 'cpu')

t_start = time.perf_counter()

results = model.track(video_path,show=True, conf=conf_threshold, save=True)

last_seen_ids = 0
for result in reversed(results):
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
        if len(ids) > 0:
            last_seen_ids = ids
            break
        
t_stop = time.perf_counter()

# print last entry in results
print("Number of melon: ", last_seen_ids)
print("Time taken to compute: ",t_stop - t_start)
