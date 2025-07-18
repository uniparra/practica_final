from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from motoria.iamodel import IaModel
from motoria.mlmodels import MlModel


df_data = pd.read_csv("./dataset_practica_final.csv")

input = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif, mutual_info_classif],
                'selector__k': [5, 10, 'all'],
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

inputMejorado = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['hotel', 'distribution_channel', 'arrival_date_year', 'arrival_date_month',
                         'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
                         'stays_in_week_nights', 'adults', 'children', 'babies', 'country', 'is_repeated_guest',
                         'reserved_room_type', 'booking_changes', 'agent', 'company', 'days_in_waiting_list', 'adr',
                         'reservation_status', 'reservation_status_date', 'meal', 'market_segment'],
        "cols_label_encod": ['assigned_room_type'],
        "cols_one_hot_encod": ['deposit_type', 'customer_type', ],
        "nan_col_mean": [],
        "nan_col_mode": []
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif],
                'selector__k': ['all'],
                'model__C': [1],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

inputBalanceado = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
    },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod": ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type',
                               'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode": ["country"]
    },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('balance', SMOTE()),
            ('model', LogisticRegression())
        ]),

        "param_grid": [
            {
                'selector__score_func': [f_classif],
                'selector__k': ['all'],
                'model__C': [1],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear']
            }
        ],
        "scoring": "accuracy"
    }
}

inputDecision = {
    "data_input": {
        "dataset": df_data,
        "target_name": "is_canceled",
        },
    "data_preproces": {
        "test_size": 0.2,
        "stratify": True,
        "cols_to_drop": ['reservation_status', 'reservation_status_date', 'company'],
        "cols_label_encod": ['arrival_date_month', 'reserved_room_type', 'assigned_room_type', 'country'],
        "cols_one_hot_encod" : ['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type', 'deposit_type'],
        "nan_col_mean": ["children", "agent"],
        "nan_col_mode" : ["country"]
        },
    "train": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest()),
            ('model', DecisionTreeRegressor())
            ]),

        "param_grid": [
                {
                    'model': [DecisionTreeClassifier()],
                    'selector__score_func': [f_classif], #Analizar la posibilidad de usar otro selector de espacio de, p. ej. SelectFromModel
                    'selector__k': [5, 10, 'all'],
                    'model__max_depth': [None, 5, 10],
                    'model__min_samples_split': [2, 5, 10]
                },
                ],
        "scoring": "accuracy"
        }
}


modeloRegresion = MlModel(input)
modeloDecision = MlModel(inputDecision)
modeloRegresionMejorado = MlModel(inputMejorado)
modeloRegresionBalanceado = MlModel(inputBalanceado)

def mostrar_resultados(modelo: MlModel):
    best_pipeline = modelo.optimized_model.best_estimator_
    mask = best_pipeline.named_steps['selector'].get_support()

    print("Resultados del modelo:")
    print(f"Mejores parámetros: {modelo.optimized_model.best_params_}")
    print(f"Mejor puntuación: {modelo.optimized_model.best_score_}")
    print(f"Mejor estimador: {modelo.optimized_model.best_estimator_}")
    print(f"Coeficientes del modelo: {modelo.optimized_model.best_estimator_.named_steps['model'].coef_}")
    print(f"Espacio de características, {list(modelo.X.columns[mask])}")
    IaModel.plot_confusion(modelo.optimized_model, modelo.X_test, modelo.y_test)
    IaModel.plot_feature_importance(modelo.optimized_model.best_estimator_['model'],
                                    list(modelo.X.columns[mask]))
    IaModel.plot_roc_curve(modelo.optimized_model, modelo.X_test, modelo.y_test)

mostrar_resultados(modeloRegresion)
print("-----------------------------------------------------")
mostrar_resultados(modeloRegresionMejorado)
print("-----------------------------------------------------")
mostrar_resultados(modeloRegresionBalanceado)
print("-----------------------------------------------------")
mostrar_resultados(modeloDecision)

