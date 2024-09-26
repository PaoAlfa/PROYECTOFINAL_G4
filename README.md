# Detección Temprana de Lesiones Cutáneas del Torso Posterior Inferior con IA

**Universidad de los Andes**  
**Curso: Aprendizaje No Supervisado**

### Integrantes del Proyecto:
- Paola Alexandra Cifuentes
- Paola Vanessa Alfaro Mora
- Daniel David Florez Thomas
- Camilo Ernesto Robayo Abello


## Resumen

Este mediante el análisis de imágenes de lesiones de piel, se desarrollará un algoritmo de clustering jerárquico aglomerativo para marcar y segmentar las áreas afectadas. Se utilizará una matriz de distancias reducida mediante PCA para realizar un clustering jerárquico. Con la matriz reducida se marcarán las áreas de la lesión cutánea mediante dos técnicas: encontrar el área que contenga un patrón diferente en la fotografía por el promedio de los colores y encontrar dos clústeres (piel sana y piel con lesión) mediante el algoritmo de KMedoides. El análisis se centrará en lesiones del torso posterior inferior. Se extraerá una muestra de 4.596 imágenes de un dataset de 401.059. Los resultados obtenidos permitirán identificar patrones asociados a diferentes tipos de lesiones y contribuirán al desarrollo de una herramienta de apoyo para clasificación de imágenes, facilitando la identificación de alertas tempranas de lesiones cutáneas.


## Estructura del Repositorio

- `notebooks/`: Incluye notebooks de Jupyter para análisis y entrenamiento de modelos.
- `documentación/`: Documentación y reportes del proyecto: propuestas teoricas y video resumen del proyecto.
- `data/`: Contiene datos brutos y procesados. [Enlace_acceso_bd](https://github.com/PaoAlfa/PROYECTOFINAL_G4/blob/Data/Enlace_acceso_bd). Al ser una base de datos pesada se debe ir a un enlace publico con carpeta de base de datos en drive.

![Lesión Cutánea en el Torso](https://drive.google.com/uc?export=view&id=1jDEJzlPRqr2xVLKGXUFiPLfs1HCBfrOi)

*Fuente: [Kaggle - ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge)*

## Uso

 Ejecuta el notebook `Preprocesamiento_de_imagenes_Detección_temprana_de_lesiones_de_piel.ipynb` para el análisis exploratorio de datos.

1. **Carga de Librerías Necesarias**: Importación de las librerías esenciales para el procesamiento de imágenes, análisis de datos, y visualización de resultados.
2. **Lectura de Imágenes**: Se utiliza el formato `hdf5` para almacenar y gestionar eficientemente el gran volumen de datos de imágenes.
3. **Almacenamiento de Nombres de Imágenes**: Lectura de las imágenes desde el archivo `hdf5` y almacenamiento de los nombres de cada caso para su posterior procesamiento.

## Algunas de las Librerías Utilizadas

El siguiente conjunto de librerías es esencial para llevar a cabo el procesamiento de imágenes y análisis de datos en este proyecto:

- **NumPy**: Utilizado para operaciones numéricas y manejo de matrices.
- **Pandas**: Facilita la manipulación y análisis de datos estructurados.
- **Matplotlib** y **Seaborn**: Utilizadas para la visualización de datos y gráficos.
- **OpenCV (cv2)**: Librería principal para el procesamiento de imágenes.
- **h5py**: Permite trabajar con archivos en formato HDF5, útil para manejar grandes volúmenes de datos como imágenes.

### **Bibliografía**
Codella, N., Nguyen, Q.-B., Pankanti, S., Gutman, D., Helba, B., Halpern, A., & Smith, J. R. (2017). Deep learning ensembles for melanoma recognition in dermoscopy images. IBM Journal of Research and Development.Recuperado de : https://doi.org/10.48550/arXiv.1610.04662
International Skin Imaging Collaboration. (2024). The ISIC 2024 Challenge Dataset: Official training and testing datasets of the ISIC 2024 Challenge. Recuperado de: https://doi.org/10.34970/2024-slice-3d Nicholas Kurtansky, Veronica Rotemberg, Maura Gillis, Kivanc Kose, Walter Reade, Ashley Chow. (2024). ISIC 2024 - Skin Cancer Detection with 3D-TBP. Recuperado de: https://kaggle.com/competitions/isic-2024-challenge
Yu, L., Chen, H., Dou, Q., Qin, J., & Heng, P.-A. (2017). Automated melanoma recognition in dermoscopy images via very deep residual networks. IEEE Transactions on Medical Imaging, Recuperado de: https://doi.org/10.1109/TMI.2016.2642839
Brinker, T. J., Hekler, A., Ittner, C., & Enk, A. H. (2016). Automated melanoma recognition in dermoscopy images: A systematic review. Journal of the American Academy of Dermatology, 74(4), 671-682.
Codella, N. C., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W., ... & Halpern, A. (2018, April). Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018) (pp. 168-172). IEEE. Recuperado de: https://arxiv.org/pdf/1710.05006
Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. nature, 542(7639), 115-118. Recuperado de: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8382232/
International Skin Imaging Collaboration. (2024). The ISIC 2024 Challenge Dataset: Official training and testing datasets of the ISIC 2024 Challenge. Recuperado de: https://doi.org/10.34970/2024-slice-3d
Kurtansky, N., Rotemberg, V., Gillis, M., Kose, K., Reade, W., & Chow, A. (2024). ISIC 2024 - Skin Cancer Detection with 3D-TBP. Recuperado de: https://kaggle.com/competitions/isic-2024-challenge
Tan, T. Y., Zhang, L., & Lim, C. P. (2020). Adaptive melanoma diagnosis using evolving clustering, ensemble and deep neural networks. Knowledge-Based Systems, 187, 104807. Recuperado de: https://www.sciencedirect.com/science/article/abs/pii/S0950705119302825
Yu, L., Chen, H., Dou, Q., Qin, J., & Heng, P.-A. (2017). Automated melanoma recognition in dermoscopy images via very deep residual networks. IEEE Transactions on Medical Imaging. Recuperado de: https://doi.org/10.1109/TMI.2016.2642839

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
_____________________________________________________________________________

# Definición de la cuadrícula y el paso marcación por clustering
tam_cuadricula = 25
paso = 5
#Se selecciona la imagen 20 por tener dos lesiones cutáneas
row = df_pixel.iloc[20]
row = row.drop('isic_id')
image = row.values.reshape(120, 120)
#Se recorre la imagen en bloques de 25x25 con paso de 5
lista_bloques = []
for y in range(0, image.shape[0] - tam_cuadricula + 1, paso):
    for x in range(0, image.shape[1] - tam_cuadricula + 1, paso):
        # Extraer la cuadrícula actual de 25x25
        cuadro = image[y:y + tam_cuadricula, x:x + tam_cuadricula]
        # Aplanar el bloque de 25x25 y agregarlo a la lista
        lista_bloques.append(cuadro.flatten())

# Crear un DataFrame a partir de los bloques extraídos
df_cuadriculas = pd.DataFrame(lista_bloques)
# Estandarizar los datos
scaler = MinMaxScaler()
df_cuadriculas_scaled = scaler.fit_transform(df_cuadriculas)
# Inicializar los índices de los medoids aleatorios
num_clusters = 2
initial_medoids = np.random.choice(df_cuadriculas_scaled.shape[0], num_clusters, replace=False)
# Implementar K-Medoides
kmedoids_instance = kmedoids(df_cuadriculas_scaled, initial_medoids)
kmedoids_instance.process()
clusters_kmedoids = kmedoids_instance.get_clusters()

# Añadir los resultados de los clusters al DataFrame
df_cuadriculas['Cluster_K_Medoides'] = -1  # Inicializa con -1 (no asignado)
for cluster_id, indices in enumerate(clusters_kmedoids):
    df_cuadriculas.loc[indices, 'Cluster_K_Medoides'] = cluster_id

# Mostrar la distribución de los clusters
print(f"Número de clusters formados: {num_clusters}")
print(df_cuadriculas['Cluster_K_Medoides'].value_counts())

image = image.astype(np.float32)
# Asignar un color a cada cluster
colores = {0: 'red', 1: 'blue'}
plt.imshow(image, cmap='gray')
frecuencia_clusters = {}
# Dibujar los bloques en el color correspondiente al cluster más frecuente
for idx, (y, x) in enumerate([(y, x) for y in range(0, image.shape[0] - tam_cuadricula + 1, paso)
                                 for x in range(0, image.shape[1] - tam_cuadricula + 1, paso)]):
    cluster = df_cuadriculas['Cluster_K_Medoides'].iloc[idx]
    # Si el bloque ya está en el diccionario, se incrementa el contador
    if (y, x) not in frecuencia_clusters:
        frecuencia_clusters[(y, x)] = {}
    if cluster not in frecuencia_clusters[(y, x)]:
        frecuencia_clusters[(y, x)][cluster] = 0
    frecuencia_clusters[(y, x)][cluster] += 1

# Para cada bloque se encuentra el cluster más frecuente
for (y, x), clusters in frecuencia_clusters.items():
    cluster_mas_frecuente = max(clusters, key=clusters.get)
    plt.gca().add_patch(plt.Rectangle((x+10, y+10), 5, 5, edgecolor=colores[cluster_mas_frecuente], facecolor='none', linewidth=1))
plt.show()
---


