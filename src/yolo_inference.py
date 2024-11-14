from ultralytics import YOLO

# load model trained on custom dataset
model = YOLO(r"training/runs/detect/train2/weights/best.pt", task="detect")
#load the model in GPU
model = model.cuda()
print(model.names)  # class names

model.predict(r"../input_videos/08fd33_4.mp4", save=True)