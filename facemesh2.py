import cv2
import mediapipe as mp

IMAGE_FILES = ['images/ethan.png']

for imageRef in IMAGE_FILES:
    image = cv2.imread(imageRef)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result = face_mesh.process(rgb_image)

    height, width, _ = image.shape
    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt = facial_landmarks.landmark[i]
            x = int(pt.x * width)
            y = int(pt.y * height)
            cv2.circle(image, (x, y), 5, (100, 100, 0), -1)

    cv2.imshow("Image", image)
    cv2.waitKey(0)