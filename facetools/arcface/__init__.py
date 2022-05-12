"""
copy and change from
"""

import os

import cv2
import numpy as np
# from tracker import Tracker
# from utils.headpose import get_head_pose
import onnxruntime as ort


current_dir = os.path.realpath(os.path.dirname(__file__))
model_path = os.path.join(current_dir, "arcface-int.onnx")

model = None
input_name = None

def get_model():
    global model, input_name
    if model is None:
        ort.set_default_logger_severity(3)  # disable warn
        model = ort.InferenceSession(model_path)
        input_name = model.get_inputs()[0].name
    return model, input_name


def face_embedding(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.resize(image, (112, 112))
    image = image.astype(np.float32)
    model, input_name = get_model()
    inputs = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
    # print(inputs.shape)
    output = model.run(None, {
        input_name: inputs
    })[0]
    return output


if __name__ == '__main__':
    img = np.random.rand(112, 112, 3)
    emb = face_embedding(img)
    print(emb.shape)
