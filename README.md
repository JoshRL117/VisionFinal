Este repositorio contiene los programas desarrollados para el proyecto final de la materia de Visión Artificial. A continuación, se detalla el significado de cada uno de los archivos .py incluidos en el repositorio.

## Descripción de Archivos

### 1. *databasetrain.py*
- *Función:* Entrena el modelo de redes neuronales para la predicción facial.
- *Detalles:* Este programa utiliza los datos procesados previamente para generar un modelo capaz de identificar rostros.

### 2. *databasetransformation.py*
- *Función:* Convierte la base de datos a un archivo .csv adecuado para el entrenamiento del modelo.
- *Detalles:* Realiza las transformaciones necesarias para estructurar los datos en un formato compatible con algoritmos de aprendizaje automático.

### 3. *imagesdatabase.py*
- *Función:* Obtiene las imágenes mediante OCR utilizando PyTesseract y las almacena en directorios específicos.
- *Requisitos:*  
  - Es necesario instalar Tesseract para que este programa funcione correctamente.  
  - Consulta el tutorial detallado en el reporte para instrucciones de instalación y configuración (Esta información se encuentra en la metodología del reporte).

### 4. *pasedelistaejecutable.py*
- *Función:* Aplicación principal del proyecto.
- *Detalles:*  
  - Predice qué alumno asistió al aula y registra su hora de llegada.  
  - La información se guarda en un archivo .txt para su posterior consulta.  

## Notas Importantes
- Asegúrate de instalar todas las dependencias requeridas, incluyendo PyTesseract, antes de ejecutar los programas.
- Consulta el reporte del proyecto para más detalles sobre la configuración y uso del sistema.
