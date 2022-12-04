import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = ['images/sai.png']
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def dist(i1, i2, face_landmarks, img):
    l1 = face_landmarks.landmark[i1]
    l2 = face_landmarks.landmark[i2]

    l1x = int(l1.x * width)
    l1y = int(l1.y * height)

    l2x = int(l2.x * width)
    l2y = int(l2.y * height)

    cv2.line(img, (l1x, l1y), (l2x, l2y), (0, 250, 0), 10)

    return ((l1x, l1y), (l2x, l2y), np.sqrt((l2x - l1x) ** 2 + (l2y - l1y) ** 2))


with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):

        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for id, face_landmarks in enumerate(results.multi_face_landmarks):

            height, width, _ = annotated_image.shape
            print(f'face_landmarks: {face_landmarks}')


            avgx = []
            avgy = []
            #left eye
            # for i in [469, 470, 471, 472]:
            #     pt = face_landmarks.landmark[i]
            #     x = int(pt.x * width)
            #     y = int(pt.y * height)
            #     avgx.append(x)
            #     avgy.append(y)
            #     #cv2.circle(annotated_image, (x, y), 5, (100, 100, 0), -1)
            # avgx = int(np.average(avgx))
            # avgy = int(np.average(avgy))
            # cv2.circle(annotated_image, (avgx, avgy), 5, (100, 100, 0), 2)

            print(dist(469, 470, face_landmarks, annotated_image))

            print(dist(1,168, face_landmarks, annotated_image))


            #right eye
            for i in [474, 475, 476, 477]:
                pt = face_landmarks.landmark[i]
                x = int(pt.x * width)
                y = int(pt.y * height)
                cv2.circle(annotated_image, (x, y), 5, (100, 0, 100), -1)


            #better iris selection


            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())

        # cv2.imwrite('images/annotated_image' + str(idx) + '.png', annotated_image)
        cv2.imshow('Face Mesh', annotated_image)
        cv2.waitKey(0)


