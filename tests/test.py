
import cv2
from facetools import detect, landmark
from facetools.arcface.utils import face_align, landmarks_68_to_5

img_path = '2.jpg'
img = cv2.imread(img_path)

faces = detect(img_path)

img_with_face = img.copy()
for i, face in enumerate(faces):
    box = face['box']
    img_with_face = cv2.rectangle(img_with_face, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

    face_img = img[face['top']:face['bottom'], face['left']:face['right'], :]
    cv2.imwrite(f'face_{i}.jpg', face_img)

    face_landmark = landmark(img, box)
    converted_face_img = face_align(img, landmark=face_landmark)
    cv2.imwrite(f'face_{i}_aligned.jpg', converted_face_img)

cv2.imwrite('face.jpg', img_with_face)

img_with_landmark = img.copy()
img_with_landmark5 = img.copy()

for face in faces:
    box = face['box']
    ret2 = landmark(img, box)
    for x, y in ret2:
        x, y = int(x), int(y)
        img_with_landmark = cv2.circle(img_with_landmark, (x, y), radius=0, color=(0, 255, 0), thickness=5)

    for x, y in landmarks_68_to_5(ret2):
        x, y = int(x), int(y)
        img_with_landmark5 = cv2.circle(img_with_landmark5, (x, y), radius=0, color=(0, 255, 0), thickness=5)

cv2.imwrite('landmark.jpg', img_with_landmark)
cv2.imwrite('landmark5.jpg', img_with_landmark5)
