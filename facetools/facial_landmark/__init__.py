"""
copy and change from https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch/blob/master/onnx_detector.py
"""

import os

import cv2
import numpy as np
# from tracker import Tracker
# from utils.headpose import get_head_pose
import onnxruntime as ort


current_dir = os.path.realpath(os.path.dirname(__file__))
model_path = os.path.join(current_dir, "slim_160_latest.onnx")

model = None
input_name = None

def get_model():
    global model, input_name
    if model is None:
        ort.set_default_logger_severity(3)  # disable warn
        model = ort.InferenceSession(model_path)
        input_name = model.get_inputs()[0].name
    return model, input_name


def crop_image(orig, bbox, detection_size):
    bbox = bbox.copy()
    image = orig.copy()
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    face_width = (1 + 2 * 0.2) * bbox_width
    face_height = (1 + 2 * 0.2) * bbox_height
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
    bbox[0] = max(0, center[0] - face_width // 2)
    bbox[1] = max(0, center[1] - face_height // 2)
    bbox[2] = min(image.shape[1], center[0] + face_width // 2)
    bbox[3] = min(image.shape[0], center[1] + face_height // 2)
    bbox = np.array(bbox).astype(np.int)
    croped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    h, w, _ = croped_image.shape
    croped_image = cv2.resize(croped_image, detection_size)
    return croped_image, ([h, w, bbox[1], bbox[0]])

def landmark(img, bbox, detection_size=(160, 160)):
    """
    return landmark (68, 2)
    """
    croped_image, detail = crop_image(img, bbox, detection_size)
    croped_image = (croped_image - 127.0) / 127.0
    croped_image = np.array([np.transpose(croped_image, (2, 0, 1))]).astype(np.float32)

    model, input_name = get_model()
    raw = model.run(None, {input_name: croped_image})[0][0]

    landmark = raw[0:136].reshape((-1, 2))
    landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
    landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]

    return landmark
