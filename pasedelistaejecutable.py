import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import joblib
import pyttsx3
import time
import datetime
from collections import Counter

# Inicializar pyttsx3
engine = pyttsx3.init()

def get_file_name():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"{today}.txt"
    return file_name

def save_message_to_file(message):
    file_name = get_file_name()  # Obtener el nombre del archivo
    try:
        with open(file_name, "a") as file:
            file.write(f"{message}\n")  # Añadir el mensaje con un salto de línea
        print(f"Mensaje guardado en {file_name}: {message}")
    except Exception as e:
        print(f"Error al guardar el mensaje: {e}")

def speak_text(text):
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    
    engine.say(text)
    engine.runAndWait()

def calculateHuMoments(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()  # Convertir a un array plano
    return hu_moments

def getRoi_Faces(img, reduction_factor=0.5):
    height, width = img.shape[:2]
    roi_height = height // 2
    reduced_width = int(width * reduction_factor)
    x_start = (width - reduced_width) // 2
    y_start = 0
    x_end = x_start + reduced_width
    y_end = roi_height
    roi = img[y_start:y_end, x_start:x_end]
    return roi

def face_image_preprocess(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    roi = getRoi_Faces(binary)
    hu_moments = calculateHuMoments(roi)
    reshaped_array = np.array(hu_moments.tolist()).reshape(1, -1)
    return reshaped_array

def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Error al abrir la cámara.")
        return

    # Cargar el modelo y el escalador
    try:
        mlp_loaded = joblib.load('./mlp_mode_Paselistal23.pkl')
        scaler_loaded = joblib.load('./scaler_paselista2.pkl')
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el modelo: {e}")
        return

    start_time = time.time()
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame. Verifica que la cámara esté disponible.")
            break

        # Procesar imagen y predecir
        face = face_image_preprocess(frame)
        new_example_scaled = scaler_loaded.transform(face)
        prediction = mlp_loaded.predict(new_example_scaled)[0]
        print(f"Prediction ->>>>>>>>>>>>>>>>>>>>>>>>>>{prediction}")
        predictions.append(prediction)
        time.sleep(0.5)

        # Mostrar la cámara
        cv2.putText(frame, "Identificando...", (50, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        # Salir si se presiona 'q' o si pasan 10 segundos
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Determinar la predicción más frecuente
    if predictions:
        most_common_prediction = Counter(predictions).most_common(1)[0][0]
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        final_message = f"                                  Buenos días {most_common_prediction}. Hora de llegada: {current_time}"
        speak_text(final_message)
        show_result_window(f"{most_common_prediction}. Hora de llegada: {current_time}")
    else:
        messagebox.showinfo("Información", "No se realizaron predicciones.")

def show_result_window(message):
    result_window = tk.Toplevel()
    result_window.title("Resultado")
    
    # Mostrar el mensaje en la ventana
    result_label = tk.Label(result_window, text=message, font=("Arial", 18), fg="green")
    result_label.pack(pady=20)
    
    # Guardar el mensaje en un archivo de texto automáticamente
    save_message_to_file(message)
    
    # Función para cerrar la ventana después de 5 segundos
    def close_after_delay():
        result_window.destroy()
    
    # Configurar el temporizador para cerrar la ventana después de 5 segundos
    result_window.after(3000, close_after_delay)
    
    # Agregar el botón de cerrar manualmente
    close_button = tk.Button(result_window, text="Cerrar", command=result_window.destroy, font=("Arial", 12))
    close_button.pack(pady=10)

# Interfaz gráfica con tkinter
ventana = tk.Tk()
ventana.title("Reconocimiento Facial con Predicciones")

# Elementos de la interfaz
label_var = tk.StringVar()
label_var.set("Presiona el botón para iniciar la cámara.")

label = tk.Label(ventana, textvariable=label_var, font=("Arial", 14))
label.pack(pady=10)

boton_iniciar = tk.Button(ventana, text="Iniciar pase de lista", command=start_camera, font=("Arial", 12))
boton_iniciar.pack(pady=20)

boton_salir = tk.Button(ventana, text="Salir", command=ventana.destroy, font=("Arial", 12))
boton_salir.pack(pady=10)

ventana.mainloop()
