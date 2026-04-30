import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/ECGNet.yaml')
    # model.load('path/to/pretrained.pt')  # optional pretrained weights
    model.train(data='ultralytics/cfg/datasets/ZGTT.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                close_mosaic=10,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='runs/ZGTT',
                name='ECGNet',
                )
