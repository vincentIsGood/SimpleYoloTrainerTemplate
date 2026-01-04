import os
from ultralytics import YOLO

latest_chkpt = os.listdir("runs/detect")
latest_chkpt.sort(key=lambda x: os.path.getmtime(f"runs/detect/{x}"))

model = YOLO(f"runs/detect/{latest_chkpt[-1]}/weights/best.pt")
results = model.predict("training/train/images/image1.png", conf=0.5)

for res in results:
    if res.boxes is None:
        continue
    res.show()
    # for box in res.boxes:
    #     print(model.names[int(box.cls.item())])