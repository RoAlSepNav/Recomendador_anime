# Recomendador Anime 🍜

<img src="https://images6.alphacoders.com/656/thumb-1920-656029.png">

## Tabla de contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto-clipboard)
2. [Herramientas Utilizadas](#herramientas-utilizadas-wrench)
3. [Estructura del Proyecto](#estructura-del-proyecto-open_file_folder)
4. [Cómo usar este proyecto](#cómo-usar-este-proyecto-question)
5. [Contenido del Jupyter notebook](#contenido-del-jupyter-notebook-page_facing_up)
6. [Modelos Utilizados](#modelos-utilizados-computer)
7. [Resultados](#resultados-bar_chart)


### Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) para realizar un sistema de recomendación simple.
En este proyecto es posible indicar el nombre de un anime y obtener 10 recomendaciones similares al título ingresado.

### Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy, Scipy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.
  

### Estructura del Proyecto :open_file_folder:
- anime.csv: Archivo CSV que contiene información sobre los animes, como sus títulos, géneros, calificaciones y más.
- rating.csv: Archivo CSV que contiene las calificaciones que los usuarios han dado a los animes.
- recomendador_anime.ipynb: Jupyter notebook que contiene la implementación del sistema de recomendación.

- 
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.
- submission.csv: Archivo CSV que contiene las predicciones para el archivo test.csv de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/titanic/data
2. Coloca los archivos CSV descargados (train.csv, test.csv) en la misma carpeta que este proyecto.
3. Abre el Jupyter notebook titanic.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre características y supervivencia.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir la supervivencia de los pasajeros.
- Evaluación del modelo: Evaluación del accuracy (exactitud) y desempeño del modelo.

### Modelos Utilizados :computer:
- Logistic Regression
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier
- Bernoulli Naive Bayes
- Gradient Boosting Classifier
- Voting Classifier

### Resultados :bar_chart:
Se evaluaron todos los modelos utilizando la métrica accuracy, y los resultados son los siguientes:

- Logistic Regression: Accuracy: 0.82
- K-Nearest Neighbors Classifier: Accuracy: 0.8
- Decision Tree Classifier: Accuracy: 0.8
- Random Forest Classifier: Accuracy: 0.79
- Support Vector Classifier: Accuracy: 0.83
- Bernoulli NB: Accuracy: 0.8
- Gradient Boosting Classifier: Accuracy: 0.82
- Voting Classifier: Accuracy: 0.84

Para el Voting Classifier se mejoró en 1% el puntaje máximo de accuracy obtenido previamente logrando un valor de 0.84.
Se observa que este modelo formado a partir de otros no hace overfitting a los datos ya que las métricas entre train y test son similares.
