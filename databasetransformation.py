import cv2
import numpy as np
import os
import pandas as pd

def getRoi_Faces(img, reduction_factor=0.5):
    height, width = img.shape[:2]
    roi_height = height // 2  # Mitad de la altura de la imagen
    reduced_width = int(width * reduction_factor)  # Reducir el ancho según el factor
    
    # Coordenadas para centrar el ancho reducido
    x_start = (width - reduced_width) // 2
    y_start = 0
    x_end = x_start + reduced_width
    y_end = roi_height

    roi = img[y_start:y_end, x_start:x_end]
    return roi

def calculateHuMoments(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()  # Convertir a un array plano
    return hu_moments

# Ruta del directorio principal
folder_dir = "C:/Users/joshu/Desktop/ProyectoFinal_Vision/test"
# Nombres de subdirectorios específicos
nombres_especificos = ["Isabella", "Joshua", "Melissa","Zain","Angel","Jorge","Cristopher"]

# Lista para almacenar los datos
data = []

# Recorrer subdirectorios y procesar los específicos
for subdir, _, archivos in os.walk(folder_dir):
    nombre_subdir = os.path.basename(subdir)  # Obtener el nombre del subdirectorio
    if nombre_subdir in nombres_especificos:
        print(f"Procesando subdirectorio: {nombre_subdir}")
        # Listar los archivos dentro del subdirectorio
        for archivo in archivos:
            ruta_completa = os.path.join(subdir, archivo)
            img = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)  # Leer imagen en escala de grises
            
            if img is None:
                print(f"No se pudo cargar la imagen: {ruta_completa}")
                continue
            
            # Binarizar la imagen
            _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            roi = getRoi_Faces(binary)  # Obtener la región de interés
            hu_moments = calculateHuMoments(roi)  # Calcular momentos de Hu
            
            # Añadir los datos al DataFrame
            data.append(hu_moments.tolist()+ [nombre_subdir])
            
            # Mostrar la región de interés
            cv2.imshow("Región de interés", roi)
            cv2.waitKey(100)

# Cerrar todas las ventanas abiertas por OpenCV
cv2.destroyAllWindows()

# Crear el DataFrame
columns = [f'HuMoment{i+1}' for i in range(7)] + ['clas']
df = pd.DataFrame(data, columns=columns)
#df.to_csv('Test_Vision_Final_Org_2.csv',index=False)
print(df)