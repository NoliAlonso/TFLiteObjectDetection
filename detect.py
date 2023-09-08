# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection

MODEL_FILENAME = 'customvisionpsemodel.tflite'
LABELS_FILENAME = 'labels.txt'

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_calculation_interval = 30  # Calculate FPS every 30 frames


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels, num_threads, threshold, max_detections):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)

        # Set the number of CPU threads for inference
        # self.interpreter.set_num_threads(num_threads)

        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


def run(model: str, camera_id: int, width: int, height: int, num_threads: int, threshold: float, max_detections: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    threshold: Prediction probability for displaying
    max_detections: Maximum number of objects to display
  """
  
    # Load the custom vision labels
  with open(LABELS_FILENAME, 'r') as f:
      labels = [label.strip() for label in f.readlines()]

  # Load the custom vision ML model 
  od_model = TFLiteObjectDetection(model, labels, num_threads, threshold, max_detections)
  
  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
   
  # Read the first frame to determine its dimensions
  success, image = cap.read()
  if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

  # Calculate the aspect ratio
  aspect_ratio = image.shape[1] / image.shape[0]

  # Resize dimensions based on the aspect ratio
  new_width = width
  new_height = int(new_width / aspect_ratio)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    # Grab the frame from the camera without decoding it
    grabbed = cap.grab()
    if not grabbed:
        sys.exit(
            'ERROR: Unable to grab frame from camera.'
        )

    # Read the grabbed frame
    success, image = cap.retrieve()
    if not success:
        sys.exit(
            'ERROR: Unable to retrieve frame from camera.'
        )

    counter += 1
    image = cv2.flip(image, 1)   

     # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Run object detection using the ObjectDetection instance
    detection_result = od_model.predict_image(pil_image)

    # Draw bounding boxes based on the detection result
    for detection in detection_result:
       bounding_box = detection['boundingBox']
       left = int(bounding_box['left'] * pil_image.width)
       top = int(bounding_box['top'] * pil_image.height)
       width = int(bounding_box['width'] * pil_image.width)
       height = int(bounding_box['height'] * pil_image.height)

       # Draw bounding box rectangle
       cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

       # Draw label
       label = f"{detection['tagName']} {detection['probability']:.2f}"
       cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
    # Calculate the FPS periodically
    if counter % fps_calculation_interval == 0:
        end_time = time.time()
        fps = fps_calculation_interval / (end_time - start_time)
        start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='customvisionpsemodel.tflite')
  parser.add_argument(
      '--cameraId', 
      help='Id of camera.',
      required=False, 
      type=int, 
      default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=512)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=512)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--threshold',
      help="Probability threshold.",
      required=False,
      type=float, 
      default=0.1)
  parser.add_argument(
      '--max_detections',
      help="Maximum number of detections.",
      required=False,
      type=int, 
      default=16)

  args = parser.parse_args()

  try:
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.threshold, args.max_detections)
  except Exception as e:
    print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
  main()
