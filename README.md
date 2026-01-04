# Yolo Model Training
A very very simple example (w/ required script & tools) to train your own Yolo model. It's very easy.

An annotation tool is already provided which I downloaded from https://github.com/LdDl/yolo-ann .
To use it locally, simply click on the `index.html` after extracting it.

Then, you need to
1. Select your image(s) for `Images`
2. Select `classes.names` for `Classes`

## Guidance
You need to:
1. Download the model `yolo11n.pt` and put it in `models/` from [here](https://huggingface.co/Ultralytics/YOLO11/tree/365ed86341e7a7456dbc4cafc09f138814ce9ff1)
2. Prepare data (images and labels) for `training/train/`, then `training/val/`
    - You can use `0_tool/yolo-ann-master/index.html` to get the `labels`
3. Modify `classes.names` and `dataset.yaml` to suit your needs
4. `python3 train.py`

## What is the example doing?
Train a YOLO model to detect `folder` icons.