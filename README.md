# CustomVision - TensorFlow Lite Object Detection

Created by exporting the customvision model as tensorflow zip file and sample code.

-- When using a Raspberry Pi with a Raspberry Pi Camera, due to OpenCV (cv2), it requires legacy camera to be enabled in the raspi-config.

## Start with a virtual environment

```
cd
python -m venv envtfliteobjectdetection
. envtfliteobjectdetection/bin/activate
```

## Clone the repo then enter the folder

```
git clone https://github.com/NoliAlonso/TFLiteObjectDetection
cd TFLiteObjectDetection
```

## Installation

```
pip install -r requirements.txt
```

## Run using:

```
python detect.py 
```

Expected performance is 1 FPS or less. Noticeably faster with the float16 model.
