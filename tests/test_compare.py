
import cv2
from facetools import detect, landmark, face_embedding
from facetools.arcface.utils import face_align, landmarks_68_to_5

def compute_sim(feat1, feat2):
    import numpy as np
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


img_path1 = 'player1.jpg'
img1 = cv2.imread(img_path1)

img_path2 = 'player2.jpg'
img2 = cv2.imread(img_path2)

faces1 = detect(img_path1)
faces2 = detect(img_path2)

face_landmark1 = landmark(img1, faces1[0]['box'])
converted_face_img1 = face_align(img1, landmark=face_landmark1)
cv2.imwrite('player1_face_aligned.jpg', converted_face_img1)

face_landmark2 = landmark(img2, faces2[0]['box'])
converted_face_img2 = face_align(img2, landmark=face_landmark2)
cv2.imwrite('player2_face_aligned.jpg', converted_face_img2)

face_emb1 = face_embedding('player1_face_aligned.jpg')
face_emb2 = face_embedding('player2_face_aligned.jpg')
print(face_emb1.shape)
print(face_emb2.shape)
sim = compute_sim(face_emb1, face_emb2)
print(sim)
