#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
import os
import argparse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5			# cls score
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45		# obj confidence

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)



class image_converter:

  def __init__(self,window_name,net,classes):
    self.image_pub = rospy.Publisher("/yolov6/camera/rgb/image_detection",Image, queue_size=1000)
    self.window_name = window_name
    self.net = net
    self.classes = classes
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/head_camera/rgb/image_raw",Image,self.callback)

  def draw_label(self,input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

  def pre_process(self,input_image, net):
	# Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
    net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
	# print(outputs[0].shape)
    return outputs

  def post_process(self,input_image, outputs):
	# Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []

	# Rows.
    rows = outputs[0].shape[1]
    
    image_height, image_width = input_image.shape[:2]

	# Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

	# Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

		# Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

			# Get the index of max class score.
            class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])
        self.draw_label(input_image, label, left, top)
    return input_image

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # Load image.
    #   frame = cv2.imread(img_path)
      frame = cv_image
      input = frame.copy()
    except CvBridgeError as e:
      print(e)

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the 
	# timings for each of the layers(in layersTimes)
	# Process image.
    detections = self.pre_process(input.copy(), self.net)
    img = self.post_process(frame.copy(), detections)
    label = 'YOLOv6xROS'
    cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
    # cv2.imshow(self.window_name, img)
    cv2.waitKey(3)

    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='models/yolov6n.onnx', help="Input your onnx model.")
  parser.add_argument('--img', default='sample.jpg', help="Path to your input image.")
  parser.add_argument('--classesFile', default='coco.names', help="Path to your classesFile.")
  args = parser.parse_args()

  # Load class names.
  model_path, img_path, classesFile = args.model, args.img, args.classesFile
  window_name = os.path.splitext(os.path.basename(model_path))[0]
  classes = None
  with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

  # Give the weight files to the model and load the network using them.
  net = cv2.dnn.readNet(model_path)

  # Initiliaze Node
  rospy.init_node('yolov6_ROS', anonymous=True)
  ic = image_converter(window_name=window_name,net=net,classes=classes)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
