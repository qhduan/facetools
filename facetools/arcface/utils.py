
import cv2
import numpy as np
from skimage import transform as trans


def landmarks_68_to_5(points):
    left_eye = [
        (points[37][0] + points[40][0]) / 2,
        (points[37][1] + points[40][1]) / 2
    ]
    right_eye = [
        (points[43][0] + points[46][0]) / 2,
        (points[43][1] + points[46][1]) / 2
    ]
    nose = points[30]
    left_mouth = points[48]
    right_mouth = points[54]

    return [
        left_eye,
        right_eye,
        nose,
        left_mouth,
        right_mouth
    ]


def face_align(img, landmark=None, image_size=[112, 112]):
    # Do alignment using landmark points
    assert len(image_size)==2
    if len(landmark) == 68:
        landmark = landmarks_68_to_5(landmark)

    landmark = np.array(landmark)
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
        src[:,0] += 8.0
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    assert len(image_size)==2
    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped
