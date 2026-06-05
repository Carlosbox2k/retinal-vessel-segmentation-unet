<img width="1550" height="396" alt="image" src="https://github.com/user-attachments/assets/2764a3c4-c0a7-48c9-aa3e-d408d2ae80bc" />

# Segmentación de Imágenes Médicas con Redes Convolucionales

A continuación se explica la arquitectura del proyecto y cómo ejecutarlo.

## Jerarquía de archivos

En esta sección se define el contenido de todas las carpetas.

### /data

Imágenes (escaneos, máscaras y segmentaciones).

#### /data/generated

Segmentaciones predichas por el modelo U-Net.

#### /data/test

Imágenes de prueba (escaneos, máscaras y segmentaciones manuales de los dos expertos).

#### /data/training

Imágenes de entrenamiento (escaneos, máscaras y segmentaciones manuales del experto).

### /models

Modelos U-Net generados en cada uno de los 5 pliegues del entrenamiento, en formato *.keras*.

### /src

Todo el código escrito.

#### /src/dataManagement

Módulos de gestión de las imágenes (carga de datos, data augmentation, padding y aplicación de máscaras).

## Módulos

En esta sección se define la función de cada módulo de la carpeta ```/src```.

### Find

Encuentra archivos o directorios dentro del proyecto (útil si se utiliza un servidor de Jupyter para ejecutar el proyecto).

### Metrics

Define las métricas y funciones de pérdida que se usan en el entrenamiento y para evaluar las predicciones (**DICE score, BCE**).

### Predict

Realiza predicciones con todos los modelos guardados, las evalúa y las guarda en la carpeta ```/data/generated```.

### Train

Entrena el modelo con validación cruzada con 5 pliegues y guarda el modelo obtenido en cada pliegue en la carpeta ```/models```.

### UnetModel

Implementación de la arquitectura U-Net.

<img alt="image" src="model.png" />

### DataGenerator

Realiza el aumento de datos (data augmentation).

### DataLoader

Carga los datos de entrada de la carpeta ```/data/training``` para los datos de entrenamiento y de ```/data/test``` para los datos de prueba.

### DataPostprocessing

Procesa las imágenes que predice el modelo (unpadding y multiplicación por máscaras).

### DataPreprocessing

Procesa las imágenes antes de usarlas como entradas del modelo (padding y multiplicación por máscaras).

### Padding

Métodos auxiliares para aplicar padding en DataPreprocessing y DataPostprocessing.

## Ejecución

En esta sección se explica cómo ejecutar el proyecto.

Antes de realizar cualquier ejecución es necesario **instalar las dependencias** del proyecto con el comando:

```bash
pip install -r requirements.txt
```

### Variables de entorno (en el .env)

- **HORIZONTAL_UNET_SIZE** y **VERTICAL_UNET_SIZE**: Resolución horizontal y vertical de las imágenes de entrada deseadas para el modelo U-Net. **Tienen que ser múltiplo de 16.**
- **DATA_AUGMENTATION_SIZE**: Cantidad de imágenes que se desean generar mediante data augmentation para entrenar el modelo.
- **WRITE_MODELS**: Determina si guardar los modelos entrenados en cada pliegue en la carpeta ```/models```. **Si se pone a True, reescribirá los modelos que ya estén en la carpeta.**
- **WRITE_IMAGES**: Determina si guardar las imágenes predichas en la carpeta ```/data/generated```. **Si se pone a True, reescribirá las imágenes que ya estén en la carpeta.**
- **SHOW_PREDICTIONS_IN_TRAINING**: Determina si mostrar las imágenes generadas en el entrenamiento.

### Entrenamiento

Para entrenar el modelo U-Net se debe ejecutar el módulo ***Train.py***. 

Los datos de entrada se cargan desde ```/data/training```. A medida que se entrene el modelo en cada pliegue, se imprimirá en consola el progreso de cada época, así como el **valor del DICE score** y de la **pérdida (error)** de cada una.

Tras el entrenamiento, se imprimen los DICE score medios obtenidos en cada pliegue. Además, si se ha determinado así en la variable de entorno **SHOW_PREDICTIONS_IN_TRAINING**, las segmentaciones predichas en cada pliegue del entrenamiento serán mostradas.

Si se ha determinado así en la variable de entorno **WRITE_MODELS**, los modelos entrenados en cada pliegue se guardarán en la carpeta ```/models```.

### Predicciones

Para realizar predicciones con los modelos guardados, se debe ejecutar el módulo ***Predict.py*** (debe haber modelos guardados en ```/models```).

Los datos de entrada se cargan desde */data/test*. Se realizan predicciones con todos los modelos de la carpeta ```/models```, y se crean las segmentaciones medias entre todas las predicciones.

Tras calcular las predicciones, se imprime el DICE score medio obtenido para cada experto, y la media entre ambos.

Si se ha determinado así en la variable de entorno **WRITE_IMAGES**, las segmentaciones predichas se guardarán en la carpeta ```/data/generated```.
