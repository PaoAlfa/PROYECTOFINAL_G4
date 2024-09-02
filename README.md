# Detección Temprana de Lesiones Cutáneas del Torso con IA

**Universidad de los Andes**  
**Curso: Aprendizaje No Supervisado**

### Integrantes del Proyecto:
- Paola Alexandra Cifuentes
- Paola Vanessa Alfaro Mora
- Daniel David Florez Thomas
- Camilo Ernesto Robayo Abello


## Resumen

Este proyecto tiene como objetivo desarrollar un algoritmo de clusterización para detectar lesiones cutáneas en imágenes del torso. El algoritmo clasificará las imágenes en dos clústeres: piel sana y piel con lesión. Para lograr esto, se utilizará una muestra de aproximadamente 10,000 imágenes de un dataset de 401,059. Se aplicarán técnicas de reducción de dimensionalidad como PCA (Análisis de Componentes Principales) y SVD (Descomposición en Valores Singulares), así como algoritmos de clusterización para extraer conclusiones demográficas y encontrar grupos de personas con mayor riesgo de contraer cáncer de piel.

## Estructura del Repositorio

- `notebooks/`: Incluye notebooks de Jupyter para análisis y entrenamiento de modelos.
- `documentos/`: Documentación y reportes del proyecto: propuestas teoricas.
- `data/`: Contiene datos brutos y procesados. [Enlace_acceso_bd](https://github.com/PaoAlfa/PROYECTOFINAL_G4/blob/Data/Enlace_acceso_bd). Al ser una base de datos pesada se debe ir a un enlace publico con carpeta de base de datos en drive.

![Lesión Cutánea en el Torso](https://drive.google.com/uc?export=view&id=1jDEJzlPRqr2xVLKGXUFiPLfs1HCBfrOi)

*Fuente: [Kaggle - ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge)*

## Uso

 Ejecuta el notebook `Preprocesamiento_de_imagenes_Detección_temprana_de_lesiones_de_piel.ipynb` para el análisis exploratorio de datos.

1. **Carga de Librerías Necesarias**: Importación de las librerías esenciales para el procesamiento de imágenes, análisis de datos, y visualización de resultados.
2. **Lectura de Imágenes**: Se utiliza el formato `hdf5` para almacenar y gestionar eficientemente el gran volumen de datos de imágenes.
3. **Almacenamiento de Nombres de Imágenes**: Lectura de las imágenes desde el archivo `hdf5` y almacenamiento de los nombres de cada caso para su posterior procesamiento.

## Librerías Utilizadas

El siguiente conjunto de librerías es esencial para llevar a cabo el procesamiento de imágenes y análisis de datos en este proyecto:

- **NumPy**: Utilizado para operaciones numéricas y manejo de matrices.
- **Pandas**: Facilita la manipulación y análisis de datos estructurados.
- **Matplotlib** y **Seaborn**: Utilizadas para la visualización de datos y gráficos.
- **OpenCV (cv2)**: Librería principal para el procesamiento de imágenes.
- **h5py**: Permite trabajar con archivos en formato HDF5, útil para manejar grandes volúmenes de datos como imágenes.

## Carga y Almacenamiento de Imágenes

El notebook empieza leyendo un archivo `hdf5` que contiene imágenes de casos clínicos de lesiones cutáneas. Las imágenes se almacenan en una lista que contiene los nombres de cada caso, lo que permite su posterior procesamiento y análisis.

### **Bibliografía**

1. Codella, N., et al. (2017). *Deep learning ensembles for melanoma recognition in dermoscopy images*. IBM Journal of Research and Development. Recuperado de: [https://doi.org/10.48550/arXiv.1610.04662](https://doi.org/10.48550/arXiv.1610.04662)
2. International Skin Imaging Collaboration. (2024). *The ISIC 2024 Challenge Dataset*. Recuperado de: [https://doi.org/10.34970/2024-slice-3d](https://doi.org/10.34970/2024-slice-3d)
3. Nicholas Kurtansky, et al. (2024). *ISIC 2024 - Skin Cancer Detection with 3D-TBP*. Recuperado de: [https://kaggle.com/competitions/isic-2024-challenge](https://kaggle.com/competitions/isic-2024-challenge)
4. Yu, L., et al. (2017). *Automated melanoma recognition in dermoscopy images via very deep residual networks*. IEEE Transactions on Medical Imaging. Recuperado de: [https://doi.org/10.1109/TMI.2016.2642839](https://doi.org/10.1109/TMI.2016.2642839)

### Ejemplo de Código

```python
# Llamado de librerías necesarias para el procesado de las imágenes con lesiones cutáneas o de piel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import h5py

# Archivo con las imágenes
with h5py.File('train-image.hdf5', 'r') as f:
    # Recorrer y mostrar todos los nombres de grupos y datasets
    lista = []
    for name in f:
        lista.append(name)
    print(lista)

---


