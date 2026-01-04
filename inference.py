import logging
from ultralytics import YOLO
logging.getLogger('ultralytics').setLevel(logging.WARNING)

## Model list: https://huggingface.co/Ultralytics/YOLO11/tree/365ed86341e7a7456dbc4cafc09f138814ce9ff1
# n for nano
model = YOLO("models/yolo11n.pt", verbose=False)

results = model.predict("https://ultralytics.com/images/bus.jpg", conf=0.5, save=False, verbose=False)

for res in results:
    if res.boxes is None:
        continue
    for box in res.boxes:
        print(model.names[int(box.cls.item())])
    res.show()


## From a private project
# latest_chkpt = os.listdir("runs/detect")
# latest_chkpt.sort(key=lambda x: os.path.getmtime(f"runs/detect/{x}"))
# model = YOLO(f"runs/detect/{latest_chkpt[-1]}/weights/best.pt")
# mouse = pynput.mouse.Controller()
"""
def get_game_controls(self):
    results = self.model.predict(
        ImageGrab.grab().crop(self.screen_center_crop_box), 
        conf=0.30, 
        verbose=False
    )
    detected_controls = []
    for res in results:
        if res.boxes is None:
            continue
        # res.show()

        for i in range(res.boxes.xyxy.shape[0]):
            classIndex = int(res.boxes.cls[i].item())
            classifiction = self.model.names[classIndex]
            box = res.boxes.xyxy[i]
            x1, y1, x2, y2 = box.tolist()
            box_center_coord = self.res_coord_to_screen_coord((
                (x1 + x2)/2, 
                (y1 + y2)/2
            ))
            detected_controls.append({
                "name": classifiction,
                "center": (
                    int(box_center_coord[0] + self.crop_box_start_coord[0]), 
                    int(box_center_coord[1] + self.crop_box_start_coord[1]), 
                ),
            })
    return detected_controls
"""