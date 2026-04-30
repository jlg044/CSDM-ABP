import csv
import os
import cv2
import mediapipe as mp

#CONFIGURA AQUÍ TUS RUTAS
ruta_csv = "C:/Users/esthe/Desktop/TFG/Código/Dataset/Datos/Isa Derecha/"  # <-- CAMBIA AQUÍ la carpeta donde están tus CSV
ruta_output = "C:/Users/esthe/Desktop/UNO"  # <-- Y AQUÍ donde quieres guardar las imágenes
os.makedirs(ruta_output, exist_ok=True)

hand_csv_path = os.path.join(ruta_csv, 'hand_keypoints.csv')
pose_csv_path = os.path.join(ruta_csv, 'pose_keypoints.csv')

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hand_keypoints_names = [
    "Muñeca", "Pulgar_Base", "Pulgar_Medio", "Pulgar_Punta", "Pulgar_Extremo",
    "Indice_Base", "Indice_Medio", "Indice_Extremo", "Indice_Punta",
    "Medio_Base", "Medio_Medio", "Medio_Extremo", "Medio_Punta",
    "Anular_Base", "Anular_Medio", "Anular_Extremo", "Anular_Punta",
    "Meñique_Base", "Meñique_Medio", "Meñique_Extremo", "Meñique_Punta"
]

pose_keypoints_names = {
    11: "Hombro_Izquierdo",
    12: "Hombro_Derecho",
    13: "Codo_Izquierdo",
    14: "Codo_Derecho",
    15: "Muñeca_Izquierda",
    16: "Muñeca_Derecha",
    23: "Cadera_Izquierda",
    24: "Cadera_Derecha"
}

def leer_keypoints(csv_path, timestamp_buscado):
    keypoints = {}
    with open(csv_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['Timestamp']) == timestamp_buscado:
                keypoints[row['Nombre_Keypoint']] = (float(row['X']), float(row['Y']))
    return keypoints

def leer_keypoints_manos(csv_path, timestamp_buscado):
    manos = {'izquierda': {}, 'derecha': {}}
    with open(csv_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['Timestamp']) == timestamp_buscado:
                mano = 'derecha' if row['Mano'] == '0' else 'izquierda'
                nombre = row['Nombre_Keypoint']
                x, y = float(row['X']), float(row['Y'])
                manos[mano][nombre] = (x, y)
    return manos

def obtener_timestamps(csv_path):
    timestamps = set()
    with open(csv_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.add(int(row['Timestamp']))
    return sorted(list(timestamps))



timestamps = obtener_timestamps(pose_csv_path)
print(f"Generando imágenes para {len(timestamps)} timestamps...")
cont = 0
for timestamp_buscado in timestamps:
    hand_keypoints_dict = leer_keypoints_manos(hand_csv_path, timestamp_buscado)
    pose_keypoints = leer_keypoints(pose_csv_path, timestamp_buscado)

    # Crear imagen blanca
    img = cv2.UMat(600, 600, cv2.CV_8UC3).get()
    img[:] = (255, 255, 255)

    # Dibujar manos
    for mano, keypoints in hand_keypoints_dict.items():
        if keypoints:
            for connection in mp_hands.HAND_CONNECTIONS:
                kp1, kp2 = connection
                if kp1 < len(hand_keypoints_names) and kp2 < len(hand_keypoints_names):
                    kp1_name = hand_keypoints_names[kp1]
                    kp2_name = hand_keypoints_names[kp2]
                    if kp1_name in keypoints and kp2_name in keypoints:
                        x1, y1 = int(keypoints[kp1_name][0] * 600), int(keypoints[kp1_name][1] * 600)
                        x2, y2 = int(keypoints[kp2_name][0] * 600), int(keypoints[kp2_name][1] * 600)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            for punto in keypoints.values():
                x, y = int(punto[0] * 600), int(punto[1] * 600)
                cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

    # Dibujar pose
    if pose_keypoints:
        pose_connections = [
            (11, 13), (13, 15),     # Brazo izquierdo
            (12, 14), (14, 16),     # Brazo derecho
            (11, 12),               # Hombros
            (11, 23), (12, 24),     # Hombro -> cadera
            (23, 24)                # Caderas
        ]
        for kp1, kp2 in pose_connections:
            if kp1 in pose_keypoints_names and kp2 in pose_keypoints_names:
                kp1_name = pose_keypoints_names[kp1]
                kp2_name = pose_keypoints_names[kp2]
                if kp1_name in pose_keypoints and kp2_name in pose_keypoints:
                    x1, y1 = int(pose_keypoints[kp1_name][0] * 600), int(pose_keypoints[kp1_name][1] * 600)
                    x2, y2 = int(pose_keypoints[kp2_name][0] * 600), int(pose_keypoints[kp2_name][1] * 600)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        for punto in pose_keypoints.values():
            x, y = int(punto[0] * 600), int(punto[1] * 600)
            cv2.circle(img, (x, y), 6, (255, 0, 0), -1)


    # Guardar imagen con nombre timestamp.png
    nombre_archivo = os.path.join(ruta_output, f"{timestamp_buscado}.png")
    cv2.imwrite(nombre_archivo, img)

    print("Imagen", cont)
    cont += 1

print("Imágenes generadas y guardadas correctamente")
