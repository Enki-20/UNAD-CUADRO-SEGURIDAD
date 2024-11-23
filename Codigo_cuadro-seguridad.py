import cv2
import os
import numpy as np
import requests
import time

# Función para enviar alertas con foto a Telegram
def send_telegram_alert(message, image):
    bot_token = "Secreto"
    chat_id = "secreto"  # Tu chat_id
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    # Guarda la imagen temporalmente
    temp_image_path = "/tmp/temp_image.jpg"
    cv2.imwrite(temp_image_path, image)
    
    # Enviar mensaje y foto
    payload = {
        "chat_id": chat_id,
        "caption": message
    }
    files = {'photo': open(temp_image_path, 'rb')}
    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            print("Alerta y foto enviadas por Telegram")
        else:
            print(f"Error al enviar alerta: {response.text}")
        files['photo'].close()
    except Exception as e:
        print(f"Error de conexión: {e}")

# Cargar el clasificador Haar y el modelo LBPH
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carga las imágenes de entrenamiento
def load_registered_faces(folder="registered_faces"):
    database = {}
    labels = []
    images = []
    label_map = {}

    for label, filename in enumerate(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            labels.append(label)
            label_map[label] = filename.split('.')[0]  # Asocia la etiqueta con el nombre
    recognizer.train(images, np.array(labels))
    return label_map

registered_faces = load_registered_faces()

# Captura de video
def restart_camera(cap):
    cap.release()  # Liberar la cámara actual
    cap = cv2.VideoCapture(0)  # Reconectar la cámara
    return cap

cap = cv2.VideoCapture(0)  # Cambiar índice si la cámara no está funcionando con 0
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

last_alert_time = 0  # Variable para almacenar el tiempo de la última alerta
alert_interval = 300  # Intervalo en segundos (5 minutos)

# Variable para indicar si ya se envió un mensaje de bienvenida
welcome_sent = False

while cap.isOpened():
    ret, frame = cap.read()
    
    # Verifica si la cámara se desconectó
    if not ret:
        print("Error al leer el frame. Intentando reiniciar la cámara.")
        cap = restart_camera(cap)  # Reiniciar la cámara
        continue  # Volver a intentar leer el frame
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta los rostros
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]

        # Reconocimiento facial usando LBPH
        label, confidence = recognizer.predict(face)
        
        if confidence < 100:  # Confianza alta, es un rostro conocido
            name = registered_faces.get(label, "Desconocido")
            label_text = f"Bienvenido: {name}"

            # Solo enviar la foto y el mensaje de bienvenida una vez
            if not welcome_sent:
                send_telegram_alert(f"¡Bienvenido, {name}!", frame)
                welcome_sent = True
        else:
            label_text = "No reconocido"
            current_time = time.time()
            
            # Solo enviar alerta si han pasado 5 minutos desde la última alerta
            if current_time - last_alert_time > alert_interval:
                send_telegram_alert("¡Alerta! Se detectó un rostro no reconocido.", frame)
                last_alert_time = current_time  # Actualiza el tiempo de la última alerta
        
        # Dibuja la caja y etiqueta
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Muestra la imagen en vivo
    cv2.imshow("Reconocimiento Facial en Vivo", frame)

    # Espera por 1 ms, y si se presiona ESC, termina el bucle
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
