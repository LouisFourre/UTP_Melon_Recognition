from ultralytics import YOLO
import sys, time, torch

t_start = time.perf_counter()

# Parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."
model_path = "models/detect/V3/weights/best.pt"

# Model parameters
conf_threshold = 0.8

model = YOLO(model_path)
model.to('cuda' if torch.cuda.is_available() else 'cpu') # Need pytorch with cuda support!!



results = model.track(video_path, show=True, conf=conf_threshold, save=True, stream=True,tracker="tracker.yaml",persist=True)

# get the last seen melon ids, cause each id is unique, last id is also the nulber of melons
last_seen_id = -1

if(model_path == "models/detect/V3/weights/best.pt"):
    class_names = model.names 
    seen_ids = {}
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            cls = result.boxes.cls.cpu().numpy().astype(int)

            for obj_id, obj_cls in zip(ids, cls):
                if obj_cls not in seen_ids:
                    seen_ids[obj_cls] = set()
                seen_ids[obj_cls].add(obj_id)

    print("Number of objects per class :")
    for obj_cls, ids in seen_ids.items():
        class_name = class_names.get(obj_cls, f"Class {obj_cls}")
        print(f"{class_name}: {len(ids)}")
else:
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            max_id_in_frame = ids.max(initial=-1)
            last_seen_id = max(last_seen_id, max_id_in_frame)
        
t_stop = time.perf_counter()

# Print results
print("Number of melon: ", last_seen_id)
print("Time taken to compute: ",t_stop - t_start)
