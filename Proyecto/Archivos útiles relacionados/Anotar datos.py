import mediapipe as mp
import cv2
import csv
import os
import time
import keyboard
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

hand_keypoints_names = [
    "Muñeca", "Pulgar_Base", "Pulgar_Medio", "Pulgar_Punta", "Pulgar_Extremo",
    "Indice_Base", "Indice_Medio", "Indice_Extremo", "Indice_Punta",
    "Medio_Base", "Medio_Medio", "Medio_Extremo", "Medio_Punta",
    "Anular_Base", "Anular_Medio", "Anular_Extremo", "Anular_Punta",
    "Meñique_Base", "Meñique_Medio", "Meñique_Extremo", "Meñique_Punta"
]

pose_keypoints_names = {
    11: "Hombro_Izquierdo", 12: "Hombro_Derecho",
    13: "Codo_Izquierdo",   14: "Codo_Derecho",
    15: "Muñeca_Izquierda", 16: "Muñeca_Derecha",
    23: "Cadera_Izquierda", 24: "Cadera_Derecha"
}

acciones_teclas = {
    'f1': 'lavarse los dientes',
    'f2': 'cepillarse el pelo',
    'f3': 'lavarse la cara',
    'f4': 'echarse colonia',
    'f5': 'tomarse las pastillas',
    'f6': 'echarse desodorante',
    'f7': 'lavarse las manos'
}

hand_csv_path = os.path.join(current_dir, 'hand_keypoints.csv')
pose_csv_path = os.path.join(current_dir, 'pose_keypoints.csv')
face_csv_path = os.path.join(current_dir, 'face_keypoints.csv')
acciones_csv_path = os.path.join(current_dir, 'acciones_teclas.csv')

with open(hand_csv_path, 'a', newline='', encoding='utf-8-sig') as hand_file, \
        open(pose_csv_path, 'a', newline='', encoding='utf-8-sig') as pose_file, \
        open(face_csv_path, 'a', newline='', encoding='utf-8-sig') as face_file, \
        open(acciones_csv_path, 'a', newline='', encoding='utf-8-sig') as acciones_file:

    hand_writer = csv.writer(hand_file)
    pose_writer = csv.writer(pose_file)
    face_writer = csv.writer(face_file)
    acciones_writer = csv.writer(acciones_file)

    
    hand_writer.writerow(['Timestamp', 'Mano', 'Nombre_Keypoint', 'X', 'Y', 'Z'])
    pose_writer.writerow(['Timestamp', 'Nombre_Keypoint', 'X', 'Y', 'Z'])
    face_writer.writerow(['Timestamp', 'Face_Keypoint_Index', 'X', 'Y', 'Z'])

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        timestamp = int(time.time() * 1000)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                for kp_idx, landmark in enumerate(hand_landmarks.landmark):
                    hand_writer.writerow([timestamp, hand_idx, hand_keypoints_names[kp_idx], landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if pose_results.pose_landmarks:
            for kp_idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if kp_idx in pose_keypoints_names:
                    pose_writer.writerow([timestamp, pose_keypoints_names[kp_idx], landmark.x, landmark.y, landmark.z])
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    face_writer.writerow([timestamp, idx, landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        cv2.imshow('MediaPipe Hands, Pose & Face Mesh', image)

        if keyboard.is_pressed('esc'):
            break

        for tecla, accion in acciones_teclas.items():
            if keyboard.is_pressed(tecla):
                acciones_writer.writerow([timestamp, accion])
                time.sleep(0.3)

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    face_mesh.close()
