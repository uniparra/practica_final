# IA Model Framework

##  Objetivo del Framework

Este proyecto tiene como objetivo construir un sistema extensible y orientado a objetos para **entrenar, evaluar y comparar modelos de Machine Learning y Deep Learning** de forma unificada y organizada. Está diseñado desde el inicio con la intención de que, en futuras etapas, **pueda conectarse fácilmente con una interfaz gráfica o API REST**, permitiendo al usuario ajustar modelos de forma sencilla.

Para facilitar esa integración, **las configuraciones de los modelos se definen como diccionarios, que en un futuro podrían ser serializables en formato `JSON`.

---

##  Diseño General

El sistema se estructura en torno a una clase base llamada `IaModel` y dos clases hijas:

- `MlModel`: para modelos clásicos de machine learning (como `LogisticRegression`, `RandomForest`, etc.).
- `DlModel`: para modelos de redes neuronales profundas (`Keras` + `Hyperopt` para tuning).

Cada clase:
- Recibe como entrada un diccionario con los parámetros necesarios (`entry`).
- Integra **todo el pipeline de procesamiento**, desde la selección de datos hasta la evaluación.
- Permite comparar distintos modelos al almacenar internamente las métricas de rendimiento.

---

##  Clases Principales

###  `IaModel` (Clase base)

Métodos principales:
- `__init__(entry)`: recibe el diccionario con la configuración.
- `preprocessing()`: separa `X` e `y`, divide `train/test`, tratamiento automático de outliers si procede.
- `evaluate()`: calcula las métricas (accuracy, f1, etc.).
- `construir_tabla_comparacion([other_model])`: compara el modelo actual con otro modelo.
- `metrics`: devuelve las métricas del modelo.
- `best_params()`: devuelve un diccionario con los mejores hiperparámetros encontrados durante el entrenamiento.

Métodos secundarios:
- `plot_confusion()`
- `plot_roc_curve()`
- `plot_feature_importance()` (solo para ML)

---

###  `MlModel` (hereda de `IaModel`)

Métodos principales:
- `train()`: configura y entrena el pipeline con `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`, `HalvingRandomizedSearch` o entrenamiento directo.
- Usa internamente:
  - `Pipeline` de Scikit-learn.
  - Selección de características (`SelectKBest`, etc.).
  - Escalado (`StandardScaler`, etc.).
  - Búsqueda de hiperparámetros si está definida.

---

###  `DlModel` (hereda de `IaModel`)

Métodos principales:
- `train()`: 
  - Realiza una división `train/val` adicional interna.
  - Calcula `class_weight` para datasets desbalanceados.
  - Lanza optimización con `Hyperopt` usando un `objective()` interno.
  - Entrena una red neuronal `Keras` con los mejores hiperparámetros.

Métodos secundarios:
- `plot_training_history()`: para visualizar el aprendizaje del modelo.
- `save_model()`, `load_model()` (pendiente de implementación futura).

---

##  Ejemplo de entrada 

###  Ejemplo de entrada para un modelo de machine learning

```python
lr_config_1 = {
    "entry_id": "lr_config_1",
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": cols_to_drop,
        "cols_label_encod": cols_label_encod,
        "cols_one_hot_encod": cols_one_hot_encod,
        "nan_col_mean": nan_col_mean,
        "nan_col_mode": nan_col_mode
    },
    "train": {
        "mode": "halving_grid_search",
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', LogisticRegression(random_state=42, n_jobs=-1))
        ]),
        "param_grid": [
            {
                'selector__k': [20, 30, 'all'],
                'model__C': [0.1, 0.5, 1],
                'model__penalty': ['l1', 'l2', 'elasticnet'],
                'model__l1_ratio': [0.25, 0.5, 0.75],
                'model__solver': ['liblinear', 'saga'],
                'model__max_iter': [500],
                'model__class_weight': [None]
            }
        ],
        "scoring": "f1",
        "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    }
}
```

###  Ejemplo de entrada para un modelo de deep learning

```python
dl_config_opt = {
    "entry_id": "dl_prueba_optimizada",
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'arrival_date_day_of_month', 'company'],
        "cols_label_encod": ['country', 'reserved_room_type', 'assigned_room_type'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type', 'deposit_type', 'arrival_date_month', 'arrival_date_year'],
        "nan_col_mean": ['children', 'agent'],
        "nan_cols_mode": ['country'],
        "outliers": False, 
        "outliers_threshold": 0.01
    },
    "train": {
        "max_evals": 20, 
        "search_space": {
            "num_hidden_layers": [1, 2], 
            "units": [64, 96, 128], 
            "learning_rate": [0.001, 0.002, 0.005, 0.01], 
            "dropout_rate": [0.0, 0.05, 0.1], 
            "batch_size": [64, 128], 
            "epochs": [30, 40, 50], 
        },
        "callbacks_config": {
            "reduce_lr": {
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 4, 
                "min_lr": 1e-6 
            },
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 8, 
                "restore_best_weights": True
            }
        }
    }
}
```