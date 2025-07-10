import sys, time, torch, cv2
from ultralytics import solutions

t_start = time.perf_counter()

# Model parameters
conf_threshold = 0.8

# Video parameters
video_path = sys.argv[1] if len(sys.argv) > 1 else "No video path provided."
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_quarter = [(int(w/4), 0), (int(w/4), int(h))]
line_middle  = [(int(w/2), 0), (int(w/2), int(h))]
line_3quarter = [(int(3*w/4), 0), (int(3*w/4), int(h))] # line counting


# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,
    region=line_3quarter,
    model="models/detect/V2/weights/best.pt",
    conf=conf_threshold,
    show_out=False,
    tracker="tracker.yaml",
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    video_writer.write(results.plot_im)  # write the processed frame.

t_stop = time.perf_counter()

print("Time taken to compute: ",t_stop - t_start)
print("Number of Melons at 1/4:", results.in_count)
cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows