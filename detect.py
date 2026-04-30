import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.plotting import colors


if __name__ == '__main__':
    model_path = Path('weights/ecgnet_demo.pt')
    source = Path('dataset/test/visible/images')
    if not source.exists():
        source = Path('dataset/test/visible')

    colors.palette = [(255, 0, 0)]  # RGB red; converted internally for OpenCV output.
    colors.n = 1

    model = YOLO(str(model_path))
    model.predict(source=str(source),
                  imgsz=640,
                  project='runs/demo',
                  name='three_images',
                  exist_ok=True,
                  show=False,
                  save=True,
                  save_frames=True,
                  use_simotm='RGBT',
                  channels=4,
                  show_labels=False,
                  show_conf=False,
                  )
