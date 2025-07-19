# Comparativa de Modelos orientados a resolver un problema de Clasificación Binaria

-<ins>Autores</ins>: ***Unai Iparragirre*** y ***Gaizka Menéndez***

Los modelos que se entrenaron y evaluaron son los siguientes:

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

La variable objetivo de este estudio es `is_canceled`, que es la que nos ayudará a determinar si un cliente cancelará o no una reserva. Analizando el resto del dataset vemos que nuestro conjunto de datos contiene columnas conteniendo valores faltantes o "Nan". Estas columnas son:

Sacamos las siguientes intuiciones de los valores nulos que hemos sacado del info():

| Nombre Variable | Estudio Preliminar |
|---|---|
| `children` | Contiene valores nulos, no son muchos por lo que podríamos completar aquellos que falten con la media o la moda. |
| `country` | Ocurre la misma situación que con children, tiene valores nulos y habría que ver qué hacemos con ellos. |
| `agent` | Tiene valores nulos. Según la descripción puede tomar valores nulos. |
| `company` | Tiene valores nulos, ocurre lo mismo que con agent. Habría que ver qué hacer en este caso puesto que son muchos los valores nulos. |

Ahora buscaremos qué variables son aquellas que mayor impacto tienen en nuestra variable objetivo. Esto se puede ver de diferentes forma como puede ser la correlacion entre las features, esto nos puede dar una aproximación inicial a la solución o ayudarnos a comprender cuales son verdaderamente útiles. Para pioder calcular esta correlación debemos de disponer de un conjuntos de datos enteramente numérico por lo que tendremos que decidir que hacemos con las variables categóricas, strings, fechas y con aquellas columnas que hemos observado que tienen valores nulos. 


 El analisis y pruebas de preprocesamiento de las variables, además del entrenamiento y evaluacion de algunos de los algoritmos que se muestran se llevó a cabo en este notebook:  
 [Explore our hands-on data exploration notebook.](docs/Pruebas_Módulo_Final_ML_y_DL_GV.ipynb)