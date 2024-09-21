from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import shutil
import csv
import os

#===================SEPARATE IMAGES=========================
# Simplemente utilizamos este codigo para el pre-procesamiento de los datos.
# Lo que hace es asignar a cada imagen a su clase de 0 a 6. 
'''# Ruta de la carpeta que contiene las imágenes
ruta_imagenes = "data/images"

# Ruta del archivo que contiene las clases
ruta_clases = "data/label.csv"

# Crear una carpeta para cada clase
def crear_carpetas_clases():
    with open(ruta_clases, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la primera fila (título)
        for idx, clase in enumerate(reader):
            clase = clase[0]  # Obtener el valor de la clase
            carpeta_clase = os.path.join(ruta_imagenes, clase)
            if not os.path.exists(carpeta_clase):
                os.makedirs(carpeta_clase)

# Mover las imágenes a las carpetas correspondientes
def mover_imagenes_a_carpetas():
    with open(ruta_clases, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la primera fila (título)
        for idx, clase in enumerate(reader):
            clase = clase[0]  # Obtener el valor de la clase
            imagen_origen = os.path.join(ruta_imagenes, f"{idx}_{clase[0]}.png")  # nombre de archivo: 0_0.png, 1_0.png, etc.
            carpeta_destino = os.path.join(ruta_imagenes, clase[0])
            shutil.move(imagen_origen, carpeta_destino)

# Llamar a las funciones para crear las carpetas y mover las imágenes
crear_carpetas_clases()
mover_imagenes_a_carpetas()
'''
#====================DATA GENERATOR========================
# Genera y configura las imagenes, noramliza. 
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    validation_split=0.2  # Separación de datos de validación
)

# images(48x48) 
data_gen_entrenamiento = datagen.flow_from_directory('data/images', batch_size=(32),
                                                      target_size=(224, 224),
                                                      shuffle=True, subset='training') 

data_gen_pruebas = datagen.flow_from_directory('data/images', batch_size=(32),
                                                target_size=(224, 224),
                                                shuffle=True, subset='validation') 
'''
for images, labels in data_gen_entrenamiento:
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i])
    break
plt.show()
'''
#=====================MODEL=========================
# Carga el modelo MobileNetV2
# Agregar la capa MobileNetV2 desde TensorFlow Hub
# Obtener las imágenes y etiquetas de los generadores de datos
class EmotionModel:
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        # Cargar el modelo MobileNetV2 preentrenado
        mobileNetV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        mobileNetV2.trainable = False

        # Crear el modelo
        model = tf.keras.Sequential([
            mobileNetV2,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(7, activation='softmax')  # 7 clases de emociones
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, data_gen_entrenamiento, data_gen_pruebas, epochs=20, batch_size=32):
        # Entrenar el modelo
        entrenamiento = self.model.fit(
            data_gen_entrenamiento,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=data_gen_pruebas
        )
        return entrenamiento

    def set_trainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable
'''
model = EmotionModel()
train = model.train(data_gen_entrenamiento, data_gen_pruebas)
'''

# Entrenar el modelo (opcional)
# Si ya tienes los generadores de datos `data_gen_entrenamiento` y `data_gen_pruebas`
# puedes entrenar el modelo llamando a la función train()
# entrenamiento = emotion_model.train(data_gen_entrenamiento, data_gen_pruebas)

# Si deseas cambiar la entrenabilidad del modelo más tarde
# puedes usar el método set_trainable
# emotion_model.set_trainable(False)  # Para congelar las capas preentrenadas
# emotion_model.set_trainable(True)   # Para permitir que se entrenen todas las capas