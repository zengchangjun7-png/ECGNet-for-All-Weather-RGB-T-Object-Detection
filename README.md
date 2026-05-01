# ECGNet for All-Weather RGB-T Object Detection

This repository provides the implementation of **ECGNet**, an RGB-T object detection framework for all-weather detection in challenging tower-site scenes.

The code is associated with the manuscript **"Physics-Guided and Entity-Centric Fusion for All-Weather RGB-T Object Detection"**, which has been submitted to ***The Visual Computer***.

## Method Overview

ECGNet is designed to improve visible-thermal object detection under low illumination, occlusion, and cross-modal degradation. The framework uses paired visible-light and thermal-infrared images as input and performs RGB-T fusion for robust target detection.

The main ideas of the method are:

- Entity co-occurrence guidance for reliable cross-modal target alignment.
- Background redundancy suppression and complementary difference preservation.
- Physics-guided adaptive fusion using illumination and occlusion cues.

This repository contains the modified Ultralytics-based implementation used for training and inference.

## Environment Setup

Python 3.8 or later is recommended. 

Create and activate a conda environment:

```bash
conda create -n ecgnet python=3.9
conda activate ecgnet
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

If you use a CUDA GPU, please install the PyTorch version that matches your CUDA driver before installing the remaining dependencies.

## Dataset Organization

The dataset should contain paired visible and infrared images. The visible and infrared images must have the same file names.

```text
dataset/
├── train/
│   ├── visible/
│   │   ├── images/
│   │   └── labels/
│   └── infrared/
│       ├── images/
│       └── labels/
├── val/
│   ├── visible/
│   │   ├── images/
│   │   └── labels/
│   └── infrared/
│       ├── images/
│       └── labels/
└── test/
    ├── visible/
    │   └── images/
    └── infrared/
        └── images/
```

For training and validation, labels follow the standard YOLO format:

```text
class_id x_center y_center width height
```

The coordinates should be normalized to `[0, 1]`. For test images, labels are not required.

## Dataset Configuration

Edit the dataset configuration file:

```text
ultralytics/cfg/datasets/ZGTT.yaml
```

Example:

```yaml
train: dataset/train/visible/images
val: dataset/val/visible/images

nc: 1
names: ["target"]
```

Only visible image paths are specified in the YAML file. The corresponding infrared images are loaded automatically from the parallel `infrared/images` directory.

## Training

The ECGNet model configuration is located at:

```text
ultralytics/cfg/models/ECGNet.yaml
```

Before training, check the paths and hyperparameters in `train.py`. The key RGB-T settings are:

```python
use_simotm="RGBT"
channels=4
```

Example training command:

```bash
python train.py
```

A typical training configuration is:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/ECGNet.yaml")

model.train(
    data="ultralytics/cfg/datasets/ZGTT.yaml",
    imgsz=640,
    epochs=300,
    batch=4,
    device="0",
    optimizer="SGD",
    use_simotm="RGBT",
    channels=4,
    project="runs/ZGTT",
    name="ECGNet",
)
```

The trained weights are saved under:

```text
runs/ZGTT/ECGNet/weights/
```

## Demo Test on Three RGB-T Images

This repository includes three paired RGB-T test images for quick reproduction of the qualitative detection results.

The demo test data should be organized as:

```text
dataset/test/
├── visible/
│   ├── 00021.png
│   ├── 02843.png
│   └── 02963.png
└── infrared/
    ├── 00021.png
    ├── 02843.png
    └── 02963.png
```

Set the model path and source path in `detect.py`:

```python
model = YOLO("weights/ecgnet_demo.pt")

model.predict(
    source="dataset/test/visible",
    imgsz=640,
    project="runs/demo",
    name="three_images",
    save=True,
    use_simotm="RGBT",
    channels=4,
    show_labels=False,
    show_conf=False,
)
```

Run:

```bash
python detect.py
```

The detection results will be saved to:

```text
runs/demo/three_images/
```

The script only needs the visible image directory as input. The paired infrared images are loaded automatically from the corresponding `infrared` directory. The demo script disables class labels and confidence scores, so the saved images show detection boxes only.

## Notes

The dataset used in this project was provided by a China Tower project. Due to project confidentiality requirements, the full dataset cannot be publicly released. This repository only provides the code and the three demonstration images used in the paper. Please use other datasets for training and validation.

For long-term accessibility and citation, this repository has been archived on Zenodo: https://doi.org/10.5281/zenodo.19941134

## Citation

If this code is useful for your research, please cite the related paper after it is published.
