# Comparativa de Modelos orientados a resolver un problema de Clasificación Binaria

<ins>Autores</ins>: ***Unai Iparragirre*** y ***Gaizka Menéndez***

## Contexto de la problemática

En el sector hotelero, la cancelación de reservas es una preocupación constante que afecta significativamente la planificación y los ingresos. Muchos hoteles y apartamentos buscan activamente formas de detectar con precisión qué clientes podrían cancelar su reserva. Aquí es donde entra nuestro estudio: hemos desarrollado y evaluado modelos de Machine Learning y Deep Learning diseñados para predecir la probabilidad de cancelación por parte del cliente, permitiendo así a los gestores hoteleros tomar decisiones bien fundamentadas y que puedan realizar correcciones o adaptaciones en funcion de los objetivos de negocio.

Los modelos que se diseñaron, entrenaron y evaluaron para este propósito son los siguientes:

* Regresión logística
* Árbol de decisión
* Random Forest
* Gradient Boosting (XGBoost, LightGBM o CatBoost)
* Red neuronal multicapa usando Keras

Los datos con los que se trabajaran son los que desglosaremos a continuación:

## Variables

| Nombre Variable                  | Descripción                                              |
| -------------------------------- | -------------------------------------------------------- |
| `hotel`                          | Tipo de hotel: City Hotel o Resort Hotel                 |
| `is_canceled`                    | Variable objetivo: 1 si fue cancelado, 0 si no           |
| `lead_time`                      | Días entre la reserva y la fecha de llegada              |
| `arrival_date_year`              | Año de llegada                                           |
| `arrival_date_month`             | Mes de llegada                                           |
| `arrival_date_week_number`       | Número de la semana del año                              |
| `arrival_date_day_of_month`      | Día del mes de llegada                                   |
| `stays_in_weekend_nights`        | Noches de fin de semana reservadas                       |
| `stays_in_week_nights`           | Noches entre semana reservadas                           |
| `adults`                         | Número de adultos                                        |
| `children`                       | Número de niños                                          |
| `babies`                         | Número de bebés                                          |
| `meal`                           | Tipo de comida reservada                                 |
| `country`                        | País de origen del cliente                               |
| `market_segment`                 | Canal de marketing (online, offline, grupos...)          |
| `distribution_channel`           | Canal de distribución (directo, TA/TO...)                |
| `is_repeated_guest`              | 1 si el cliente ha estado anteriormente                  |
| `previous_cancellations`         | Nº de cancelaciones anteriores                           |
| `previous_bookings_not_canceled` | Nº de reservas previas no canceladas                     |
| `reserved_room_type`             | Tipo de habitación reservada                             |
| `assigned_room_type`             | Tipo de habitación asignada                              |
| `booking_changes`                | Nº de cambios en la reserva                              |
| `deposit_type`                   | Tipo de depósito: No Deposit, Refundable, etc.           |
| `agent`                          | ID del agente (puede ser nulo)                           |
| `company`                        | ID de la empresa (puede ser nulo)                        |
| `days_in_waiting_list`           | Días en lista de espera                                  |
| `customer_type`                  | Tipo de cliente: Transient, Group, etc.                  |
| `adr`                            | Average Daily Rate (precio promedio por noche)           |
| `required_car_parking_spaces`    | Plazas de parking solicitadas                            |
| `total_of_special_requests`      | Nº de peticiones especiales                              |
| `reservation_status`             | Estado final de la reserva: Check-Out, Canceled, No-Show |
| `reservation_status_date`        | Fecha en que se actualizó el estado                      |

La variable objetivo de este estudio es `is_canceled`, que es la que pretendemos predecir a partir de una serie de inputs para determinar si un cliente cancelará o no una reserva de hotel.

Sacamos la siguiente información sobre los tipos de columnas presentes en el dataset (tipo de los datos que contienen int, float, object ...). Esto nos permitirá ver que columnas presentan valores nulos en sus filas además de aquellas que puedan requerir algun procesamiento particular dependiendo del formato en el que se encuentren. Durante la fase de análisis exploratorio de datos (EDA), se identificaron columnas con valores faltantes (NaN). Estas son:

| Nombre Variable | Estudio Preliminar |
|---|---|
| `children` | Contiene valores nulos, no son muchos por lo que podríamos completar aquellos que falten con la media o la moda. |
| `country` | Ocurre la misma situación que con children, tiene valores nulos y habría que ver qué hacemos con ellos. |
| `agent` | Tiene valores nulos. Según la descripción puede tomar valores nulos. |
| `company` | Tiene valores nulos, ocurre lo mismo que con agent. Habría que ver qué hacer en este caso puesto que son muchos los valores nulos. |

Vamos a tratar una por una:

* ***Children***: Para tratar los valores faltantes de esta feature se contemplo completar los nulos con aquellos valores con la moda de la misma variable. también podríamos haber empleado la media, pero al ser una columna de valores enteros se entiende que tiene mas sentido aplicar la primera forma. Esta variable ya contiene un tipo de valor numérico (float64) por lo que no requiere ningun otro procesamiento adicional.

* ***country***: Aquí ocurre un poco lo mismo que para la feature 'children', se optó por emplear la moda para completar aquellos registros sin un Country asignado. En este caso, la feature era una columna categórica (object). Para este caso primero se aplico la moda para disponer de todos los registros rellenos y postereiormente, aplicamos una técnica de Label_encoding. Esto se decidio así porque era una columna que podía tomar cerca de dos centenas de valaores diferentes. Al convertirlos a un valor numérico comprendido en 0 <= x >≅ 180 es una varible más manejable por nuestros modelos durante el entrenamiento.

* ***agent***: para este caso inicialmente optamos por completar los valores nulos con la media de la columna. Ahora viendo otras posibilidades y haberlo reflexionado mejor, esta característica deberíamos haberla tratado de otra manera puesta que no fue óptimo nuestro tratamiento para este caso. Esta variable ya contiene un tipo de valor numérico (float64) por lo que no requiere ningun otro procesamiento adicional.

* ***company***: esta feature contiene un gran porcentaje de valores nulos. Inicialmente se optó por crear una nueva feature `has_company` indicando si pertenecía a una compañía (1) o no (0), también actuando como booleano pero dejandolo reflejado mediante un valor int64.

El resto de features se ha seguido una forma de procesar similar a las anteriores explicadas pero al no disponer de valores nulos su tratamiento se limitaba a su transformación a valores numéricos para identificar las variables con mayor impacto en la variable objetivo mediante diferentes metodologías que se basaban por ejemplo en un análisis de correlación. El tratamiento de todas la features del dataset se puede observar a continuación en varios notebook empleados para pruebas de diferente tipo.


 El analisis y pruebas de preprocesamiento de las variables, además del entrenamiento y evaluacion de algunos de los algoritmos que se muestran se llevó a cabo en estos dos notebook:  
 [Explore Gaizka's hands-on data exploration notebook.](docs/Pruebas_Módulo_Final_ML_y_DL_Neural_Net_Version.ipynb)

 [Explore Unai's hands-on data exploration notebook.](https://colab.research.google.com/drive/1pkE3AoVObuamCMI93yGLzf1mg1HBnpzC?authuser=1#scrollTo=BDPKObigZi_M)
 


[Eleccion final y comparación de resultados](./Elección%20final.png)
## Conclusiones

 Tras probar con diferentes modelos y distintas configuraciones, hemos llegado a las siguientes conclusiones:
* Importancia de la selección y transformación de las features.
  * Cuando incluimos arrival_ralation = [before, after, equal], como indicador de si había habido un reservtion_status modificado previo al día de llegada o no, creamos sin quererlo una variable trivial, ya que casi exactamente el mismo comportamiento que is canceled.
  * Al hacer validación cruzada, hemos visto que los modelos ganadores tienden a quedarse con todas las features. Y si forzamos a que se queden con menos, la precisión baja mucho.
  * El one-hot encoding de las variables categóricas es una técnica que mejora la precisión de los modelos, pero también aumenta la complejidad y el tiempo de entrenamiento.
  * Las features con muchos nulos, como `agent` y `company`, pueden ser problemáticas. En nuestro caso, al ser muchas las filas con valores nulos, decidimos eliminarlas. Sin embargo, en otros casos, podríamos optar por imputar estos valores o crear una categoría especial para ellos.

* Importancia de la validación cruzada para encontrar el mejor modelo.
  * Hemos probado diferentes técnicas de validación cruzada, StratifiedKFold, para intentar que nuestros modelos generalizan bien.
  * La validación cruzada con GridSearchCV y variantes nos han servido para encontrar la mejor configuración para cada modelo, así hemos podido partir de espacios de búsqueda muy amplios con randomSearch y luego ir acotando hasta quedarnos con uno muy bien ajustado.
  * La métrica de accuracy es importante, pero no es la única. En nuestro caso, también consideramos el recall y la F1-score para evaluar el rendimiento de los modelos ya que consideramos que para el negocio hotelero es más importante predecir bien una cancelación.
  * No cuesta obtener un buen resultado con redes neuronales, pero resulta muy difícil optimizarlo consistenemente. Para este caso, consideramos que los modelos de machine learning tradicionales son más efectivos, siendo además más fáciles de interpretar y ajustar.

* Entrenamiento de los modelos de DL probados.
Todas las técnicas que se mencionan a continuación a pesar de suponer una mayor cantidad de tiempo en su entrenamiento y en algun caso poder haber llegado a ser excesivo, han contribuido positivamente a los resultados obtenidos reforzando puntos o métricas donde los modelos de ML necesitaban un refuerzo.

  * Inicialmente tuvimos que lidiar con el desequilibrio de clases puesto que la clase minoritaria is_canceled positiva era muy inferior a la no cancelación. Para ello, exploramos dos métodos: la ponderación de las clases mediante class_weights, que asigna una mayor importancia a los ejemplos de la clase minoritaria durante el entrenamiento, y el uso de SMOTE (Synthetic Minority Over-sampling Technique) para generar nuevos ejemplos sintéticos de la clase minoritaria.
  * Al disponer de una gran cantidad de características tuvimos que dedicar tiempo a decidir la selección óptima de las mismas. Probamos combinando métodos estadísticos como SelectKBest, mutual_info_classif para identificar las características más relevantes, y el poder predictivo de modelos de ensemble como Random Forest Classifier para validar la importancia de las variables respecto al objetivo que deseabamos predecir. Finalmente resultó que la inclusión de todas las características, aunque mas costoso, fue la elección optima para obtener los mejores resultados. Esto concluye que o todas son necesarias o podríamos haber procedido de otra manera en el tratamiento de las feature originales y haber profundizado mejor en la etapa de feature engineering.
  * Respecto a la arquitectura de estas redes, experimentamos con diferentes configuraciones de capas densas y, para combatir el overfitting, implementamos técnicas de regularización como Dropout y L2.
  * La fase final fue la de tuning de hiperparámetros, donde pasamos de un enfoque de búsqueda aleatoria (RandomizedSearchCV), pasando despues por una búsqueda exhaustiva empleando GridSearchV, y terminando con un método más sofisticado y eficiente como Hyperopt, que utiliza la optimización bayesiana para encontrar los mejores parámetros de forma inteligente [HyperOpt Documentation](https://hyperopt.github.io/hyperopt/). Durante el entrenamiento, empleamos funciones callbacks como EarlyStopping para garantizar que el modelo no se sobreajustara y ReduceLROnPlateau para afinar la convergencia, buscando un proceso de optimización robusto y sistemático.


## Mejoras

* **Integración de Modelos**: Explorar la posibilidad de combinar modelos (ensemble learning) para mejorar la precisión y robustez de las predicciones.
* **Implementación de un Pipeline de Producción**: Desarrollar un pipeline de producción que permita la actualización continua de los modelos con nuevos datos y la implementación de un sistema de monitorización para detectar posibles degradaciones en el rendimiento del modelo.
* **Integración con Mlflow**: Implementar un sistema de seguimiento y gestión de experimentos utilizando MLflow para facilitar la reproducibilidad y el seguimiento de los modelos entrenados.
* **Implementación API REST**: Desarrollar una API REST que permita a los usuarios interactuar con los modelos de manera sencilla, facilitando la integración con aplicaciones externas y la implementación de un sistema de recomendación en tiempo real.
* **Parser de JSON**: Implementar un parser que permita la carga de configuraciones de modelos desde archivos JSON, facilitando la integración con interfaces gráficas y la personalización de los modelos por parte de los usuarios.
* **Estudio de overfitting**: Realizar un estudio más profundo sobre el overfitting en los modelos de machine learning y deep learning.


   
