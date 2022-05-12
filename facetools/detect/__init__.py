import os

import cv2
import onnxruntime as ort
import numpy as np

import io
import contextlib

from .box_utils import predict

# import gradio as gr

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-320 onnx model
# os.system("wget https://github.com/AK391/models/raw/main/vision/body_analysis/ultraface/models/version-RFB-320.onnx")
current_dir = os.path.realpath(os.path.dirname(__file__))
face_detector_onnx = os.path.join(current_dir, "version-RFB-640.onnx")

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])

face_detector = None

def get_face_detector():
    global face_detector
    if face_detector is None:
        ort.set_default_logger_severity(3)  # disable warn
        face_detector = ort.InferenceSession(face_detector_onnx)
    return face_detector


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num


# face detection method
def faceDetector(orig_image, threshold=0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (320, 240))
    # 640*480
    image = cv2.resize(image, (640, 480))

    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    face_detector = get_face_detector()
    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0],
                                   confidences, boxes, threshold)
    return boxes, labels, probs


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void

def detect(input_path, threshold=0.7):
    orig_image = cv2.imread(input_path)
    boxes, _, probs = faceDetector(orig_image, threshold=threshold)

    rets = []
    for box, prob in zip(boxes, probs):
        box = box.tolist()
        rets.append({
            'box': box,  # left top right bottom
            'left': box[0],
            'top': box[1],
            'right': box[2],
            'bottom': box[3],
            'width': box[2] - box[0],
            'height': box[3] - box[1],
            'prob': float(prob)
        })
    return rets


def inference(input_path, output_path='out.png', threshold=0.7, color=(255, 128, 0)):
    faces = detect(input_path, threshold=threshold)
    orig_image = cv2.imread(input_path)
    for face in faces:
        box = scale(face['box'])
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
    cv2.imwrite(output_path, orig_image)
    return output_path
