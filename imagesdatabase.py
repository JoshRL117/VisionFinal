
#ESTE ES EL BUENO WEY LOS OTROS NO SIRVEN
#####################################################################
#Aqui es donde se crea la base de datos con las imagenes 
#Los otros son de pruebas 
#Este ponlo en el proyecto main no la vayas a regar 
#THIS IS THE GOOD ONE ASSHOLE, PLEASE ADD COMMENTS TO YOUR CODE A BIG PROYECT LIKE THIS ONE 

import numpy as np
import os 
import cv2
import pytesseract
import re
import time
def getRoi(img):
    height, width = img.shape[:2]
    roi_height = int(height * 0.5)
    roi_width = width // 3
    x_start = (width - roi_width) // 2
    y_start = (height - roi_height) // 2
    x_end = x_start + roi_width
    y_end = y_start + roi_height
    roi= img[y_start:y_end, x_start:x_end]
    #cv2.imshow("Test",roi)
    return roi

def getText(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale_factor = 5 
    img_resized = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    img_inverted = cv2.bitwise_not(img_resized)
    _, thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(thresh)
    for p in regions:
        x, y, w, h = cv2.boundingRect(p)
        if 3 < w / h < 10:  # Filter by aspect ratio
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
    text = pytesseract.image_to_string(thresh, config='--psm 6 --oem 3')  # Use psm 6 and oem 3 for better OCR
    print("Extracted Text:", text)
    return text

def getName(text):
    patron_nombre = r'\b[A-Z][a-zA-Z]{2,11}\b'
    nombres = re.findall(patron_nombre,text)
    if nombres:
        nombre = nombres[0]
        print("Nombre seleccionado:", nombre)
        return nombre
    else:
        print("No se encontró un nombre.")
        return False

def getdirectory(img):
    roi=getRoi(img)
    text=getText(roi)
    return getName(text)
############################### Code Example  
impath = "test/MelitestDB.jpg"
img = cv2.imread(impath)
##############################

folder_dir = "C:/Users/joshu/Desktop/ProyectoFinal_Vision/test"
files=os.listdir(folder_dir)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

#ESTE ES EL BUENO WEY LOS OTROS NO SIRVEN
start_time = time.time()
photo_count = 48
directorio_creado = False
nombre_directorio = ""
esperar_para_directorio = 10 
while time.time() - start_time < 60:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break
    #cv2.imshow('Camera', frame)
    elapsed_time = int(time.time() - start_time)
    name=getdirectory(frame)
    if name in files:
        subdir = "C:/Users/joshu/Desktop/ProyectoFinal_Vision/test/" + name
        filename = os.path.join(subdir, f"templatee {name} {photo_count + 1}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Foto {photo_count + 1} tomada y guardada como {filename} en {name}")
        photo_count += 1
    time.sleep(1)
    # IMPORTANTE NO BORRAR PORQUE SI NO EL CÓDIGO NO ARRANCA
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
