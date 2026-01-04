from ultralytics import YOLO

## Notes on Extending the model
# > if you want to retain all classes
# You need to train the whole thing again (aka. COCO data + your custom data)
# Ref: https://community.ultralytics.com/t/finetuning-model/1252

if __name__ == '__main__':
    # Low samples to train on in this case -- more epochs (may overfit, whatever)
    model = YOLO("models/yolo11n.pt", verbose=False)
    results = model.train(data="training/dataset.yaml", epochs=500, verbose=True)
